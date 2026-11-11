"""Trantor Antiquity (Engine 8), tested offline against synthetic bare-earth grids
and mocked 3DEP, MRDS, Open Context, and BLM payloads.

The payload shapes copy live responses: exportImage BSQ (float32 grid, then a
one-bit validity mask), MRDS WFS GML 3.1.1, the MRDS record JSON, Open Context
geo-records, and the BLM Surface Management Agency query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from conftest import Conn
from numpy.typing import NDArray
from test_phase1_adapters import NOW, Router, counter, ollama, options

from spec1_bridge.adapters.archaeology_lidar import (
    BLM_SMA,
    ELEVATION_SERVICE,
    GRID_CELLS,
    MINING_SCORE,
    MODE_A,
    MODE_B,
    MRDS_RECORD,
    MRDS_WFS,
    OPEN_CONTEXT,
    ArchaeologyLidarAdapter,
    MineralDetail,
    SurveyArea,
    detect_features,
    finest_lidar,
    land_status,
    mineral_detail,
    mineral_phrase,
    nearest_mining_record,
    parse_bsq,
    parse_mrds_gml,
    to_bsq,
)
from spec1_bridge.contracts import public_interest
from spec1_bridge.mycroft_triage import run_adapter, triage_packets

AREA = SurveyArea("Jacksonville Placer Hill", 42.3115, -122.9725, "Jackson", "Oregon")
N = GRID_CELLS
ROWS, COLS = np.mgrid[0:N, 0:N].astype(np.float64)


def hillside(seed: int = 7) -> NDArray[np.float64]:
    """A slope of about 22 degrees, rising to the south, with 3 cm of noise."""
    noise = np.random.default_rng(seed).normal(0.0, 0.03, (N, N))
    grid: NDArray[np.float64] = 500.0 + 0.4 * ROWS + noise
    return grid


def flat(seed: int = 7) -> NDArray[np.float64]:
    grid: NDArray[np.float64] = 300.0 + np.random.default_rng(seed).normal(
        0.0, 0.03, (N, N)
    )
    return grid


def with_ditch(dem: NDArray[np.float64], along: str = "contour") -> NDArray[np.float64]:
    carved = dem.copy()
    if along == "contour":  # across the slope, level, as a water ditch runs
        carved[(np.abs(ROWS - 128) <= 1) & (COLS > 40) & (COLS < 200)] -= 0.8
    else:  # straight downhill, as a gully runs
        carved[(np.abs(COLS - 128) <= 1) & (ROWS > 40) & (ROWS < 200)] -= 0.8
    return carved


def with_bench(dem: NDArray[np.float64]) -> NDArray[np.float64]:
    cut = dem.copy()
    cut[(ROWS >= 120) & (ROWS < 128) & (COLS > 40) & (COLS < 200)] = 500.0 + 0.4 * 124
    return cut


def with_mound(dem: NDArray[np.float64]) -> NDArray[np.float64]:
    raised: NDArray[np.float64] = dem + 1.5 * np.exp(
        -((ROWS - 128) ** 2 + (COLS - 128) ** 2) / (2 * 4.0**2)
    )
    return raised


def columnar_basalt() -> NDArray[np.float64]:
    """Column tops in an offset (hexagonal) lattice, 14 m apart."""
    dem = flat()
    for row in range(20, N - 20, 14):
        offset = 7 if (row // 14) % 2 else 0
        for col in range(20 + offset, N - 20, 14):
            dem += 0.9 * np.exp(-((ROWS - row) ** 2 + (COLS - col) ** 2) / (2 * 2.5**2))
    return dem


def moraine() -> NDArray[np.float64]:
    ridge: NDArray[np.float64] = flat() + 5.0 * np.exp(
        -((COLS - 128) ** 2) / (2 * 9.0**2)
    ) * ((ROWS > 20) & (ROWS < 236))
    return ridge


# --- Payloads --------------------------------------------------------------------

CATALOG = {
    "features": [
        {
            "attributes": {
                "OBJECTID": 5088,
                "Name": "n43w124",
                "ProductName": "USGS_3DEP",
                "LowPS": 10.3,
                "Category": 1,
            }
        },
        {
            "attributes": {
                "OBJECTID": 58255,
                "Name": "OR_2022Jackson_C25",
                "ProductName": "USGS_3DEP",
                "LowPS": 1,
                "Category": 1,
            }
        },
        {
            "attributes": {
                "OBJECTID": 143737,
                "Name": "Ov_i02.tif",
                "ProductName": "",
                "LowPS": 75,
                "Category": 2,
            }
        },
    ]
}
COARSE_ONLY = {"features": CATALOG["features"][:1]}


def gml(*sites: tuple[str, str, str, float, float, str]) -> bytes:
    members = "".join(
        f"""
    <gml:featureMember>
      <ms:mrds gml:id="mrds.{dep_id}">
        <ms:geometry><gml:Point srsName="EPSG:4326"><gml:pos>{lat} {lon}</gml:pos></gml:Point></ms:geometry>
        <ms:dep_id>{dep_id}</ms:dep_id>
        <ms:site_name>{name}</ms:site_name>
        <ms:dev_stat>{status}</ms:dev_stat>
        <ms:url>https://mrdata.usgs.gov/mrds/show-mrds.php?dep_id={dep_id}</ms:url>
        <ms:code_list> {codes}</ms:code_list>
      </ms:mrds>
    </gml:featureMember>"""
        for dep_id, name, status, lat, lon, codes in sites
    )
    return (
        "<?xml version='1.0' encoding=\"UTF-8\" ?>"
        '<wfs:FeatureCollection xmlns:ms="http://mapserver.gis.umn.edu/mapserver" '
        'xmlns:gml="http://www.opengis.net/gml" xmlns:wfs="http://www.opengis.net/wfs">'
        f"{members}</wfs:FeatureCollection>"
    ).encode()


PLACER = (
    "10042888",
    "Jacksonville Placer 5",
    "Past Producer",
    42.3135,
    -122.9705,
    "AU",
)
GRAVEL = ("10054800", "Gravel Pit", "Past Producer", 42.3116, -122.9726, "SDG")
FAR_GOLD = ("10249437", "Pearce Mine", "Past Producer", 42.40, -122.90, "AU")


def mrds_record(dep_id: str = "10042888", **deposits: Any) -> dict[str, Any]:
    return {
        "id": f"https://mrdata.usgs.gov/mrds/record/{dep_id}",
        "type": "Feature",
        "properties": {
            "deposits": {
                "dep_id": dep_id,
                "dev_st": "Past Producer",
                "dep_tp": "Placer",
                "disc_yr": "1851",
            }
            | deposits,
            "districts": {"line": "1", "district": "Jacksonville"},
            "commodity": [
                {"commod": "Gold", "code": "AU", "import": "Primary"},
                {"commod": "Silver", "code": "AG", "import": "Tertiary"},
            ],
        },
    }


def open_context(*records: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(records)}


OC_SITE = "https://opencontext.org/subjects/3a130709-655e-4bad-8be8-407b81442e6e"


def oc_record(
    lat: float = 42.3118, lon: float = -122.9727, remains: bool = False
) -> dict[str, Any]:
    return {
        "label": "Survey locus 12",
        "oc-api:human-remains-related": remains,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"href": OC_SITE},
    }


def router(
    dem: NDArray[np.float64],
    *,
    catalog: dict[str, Any] = CATALOG,
    sites: tuple[tuple[str, str, str, float, float, str], ...] = (PLACER,),
    record: dict[str, Any] | None = None,
    archive: dict[str, Any] | None = None,
    land: str = "BLM",
) -> Router:
    return Router(
        {
            f"{ELEVATION_SERVICE}/query": catalog,
            f"{ELEVATION_SERVICE}/exportImage": httpx.Response(
                200,
                content=to_bsq(dem),
                headers={"content-type": "application/octet-stream"},
            ),
            MRDS_WFS: httpx.Response(
                200,
                content=gml(*sites),
                headers={"content-type": "text/xml; subtype=gml/3.1.1"},
            ),
            MRDS_RECORD: record if record is not None else mrds_record(),
            OPEN_CONTEXT: archive if archive is not None else open_context(),
            OC_SITE: httpx.Response(
                200,
                text="<html>Survey locus 12</html>",
                headers={"content-type": "text/html"},
            ),
            BLM_SMA: {"features": [{"attributes": {"ADMIN_AGENCY_CODE": land}}]},
        }
    )


def harvest(tmp_path: Path, mock: Router, areas: list[SurveyArea] | None = None) -> Any:
    adapter = ArchaeologyLidarAdapter(areas or [AREA], **options(tmp_path, mock))
    return adapter.harvest(counter())


def kinds(dem: NDArray[np.float64]) -> set[str]:
    return {item.kind for item in detect_features(dem).features}


# --- Canopy-stripped ditches and benches -------------------------------------------


def test_contour_ditch_is_detected() -> None:
    [ditch] = detect_features(with_ditch(hillside())).features
    assert ditch.kind == "contour_ditch"
    assert 150 <= ditch.length_m <= 165 and 2.0 <= ditch.width_m <= 4.0
    assert ditch.relief_m >= 0.5
    assert ditch.along_fall <= 0.05
    assert ditch.slope_divergence_deg > 15


def test_downhill_channel_is_drainage_not_a_ditch() -> None:
    detection = detect_features(with_ditch(hillside(), along="downhill"))
    assert detection.features == ()
    assert detection.rejected == ("a channel running downhill, as drainage does",)


def test_terrace_bench_is_detected_with_its_slope_break() -> None:
    features = detect_features(with_bench(hillside())).features
    bench = next(item for item in features if item.kind == "terrace_bench")
    assert bench.length_m >= 150 and bench.width_m >= 3.0
    assert bench.slope_divergence_deg > 15
    # A cut-and-fill bench has a cut bank above it and a fill edge below.
    assert {"contour_ditch", "linear_embankment"} <= {item.kind for item in features}


def test_mound_on_flat_ground_is_detected() -> None:
    [mound] = detect_features(with_mound(flat())).features
    assert mound.kind == "mound" and mound.relief_m > 1.0


@pytest.mark.parametrize("dem", [hillside(), flat()], ids=["hillside", "flat"])
def test_plain_ground_has_no_features(dem: NDArray[np.float64]) -> None:
    assert detect_features(dem).features == ()


def test_grid_edges_are_not_read() -> None:
    dem = hillside()
    dem[3:6, 40:200] -= 0.8  # a ditch-shaped dip inside the border
    assert detect_features(dem).features == ()


def test_mostly_empty_grid_is_not_read() -> None:
    dem = with_ditch(hillside())
    dem[:40, :] = np.nan
    detection = detect_features(dem)
    assert detection.features == () and "no elevation" in detection.rejected[0]


def test_bsq_round_trip_keeps_the_validity_mask() -> None:
    dem = hillside()
    dem[10, 10] = np.nan
    parsed = parse_bsq(to_bsq(dem))
    assert np.isnan(parsed[10, 10])
    assert np.allclose(parsed[~np.isnan(parsed)], dem[~np.isnan(dem)], atol=1e-3)
    with pytest.raises(ValueError):
        parse_bsq(b"short")


def test_only_lidar_at_a_metre_is_read() -> None:
    finest = finest_lidar(CATALOG["features"])
    assert finest is not None and finest["Name"] == "OR_2022Jackson_C25"
    assert finest_lidar(COARSE_ONLY["features"]) is None


def test_area_without_lidar_makes_no_request_for_elevation(tmp_path: Path) -> None:
    mock = router(with_ditch(hillside()), catalog=COARSE_ONLY)
    assert harvest(tmp_path, mock).packets == []
    assert mock.calls_to(f"{ELEVATION_SERVICE}/exportImage") == []


# --- Geological discrimination ------------------------------------------------------


def test_columnar_basalt_is_rejected() -> None:
    detection = detect_features(columnar_basalt())
    assert detection.features == ()
    assert "columnar basalt jointing" in detection.rejected[0]


def test_glacial_moraine_is_rejected() -> None:
    detection = detect_features(moraine())
    assert detection.features == ()
    assert detection.rejected == ("relief on a ridge of glacial moraine scale",)


def test_a_single_mound_is_not_a_lattice() -> None:
    assert "mound" in kinds(with_mound(flat()))


def test_basalt_makes_no_lead_even_beside_a_gold_placer(tmp_path: Path) -> None:
    assert harvest(tmp_path, router(columnar_basalt())).packets == []


# --- Historic gold placer records ----------------------------------------------------


def test_mrds_gml_is_read() -> None:
    [placer, gravel] = parse_mrds_gml(gml(PLACER, GRAVEL))
    assert (placer.dep_id, placer.codes, placer.dev_status) == (
        "10042888",
        ("AU",),
        "Past Producer",
    )
    assert placer.latitude == 42.3135 and gravel.codes == ("SDG",)
    with pytest.raises(ValueError):
        parse_mrds_gml(b"not xml")


def test_nearest_record_must_be_a_mining_commodity_within_a_kilometre() -> None:
    found = nearest_mining_record(AREA, parse_mrds_gml(gml(GRAVEL, PLACER, FAR_GOLD)))
    assert found is not None
    record, gap = found
    assert record.dep_id == "10042888" and gap < 1.0
    assert nearest_mining_record(AREA, parse_mrds_gml(gml(GRAVEL, FAR_GOLD))) is None


def test_mineral_detail_reads_commodity_and_dates() -> None:
    detail = mineral_detail(mrds_record())
    assert detail == MineralDetail(
        "gold", "placer", "Past Producer", 1851, (), "Jacksonville"
    )
    assert mineral_phrase(detail) == "a historic gold placer discovered in the 1850s"


@pytest.mark.parametrize(
    ("deposits", "phrase"),
    [
        ({"yr_fst_prd": "1884"}, "a historic gold placer active in the 1880s"),
        (
            {"disc_yr": "", "dep_tp": "Vein", "dev_st": "Prospect"},
            "a historic gold prospect, with no dated activity",
        ),
        ({"dep_tp": "Vein"}, "a historic gold mine discovered in the 1850s"),
    ],
)
def test_mineral_phrase_says_only_what_is_dated(
    deposits: dict[str, Any], phrase: str
) -> None:
    assert mineral_phrase(mineral_detail(mrds_record(**deposits))) == phrase


def test_ditch_beside_a_gold_placer_is_a_mining_lead(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()))).packets

    assert packet.adapter_name == "trantor_antiquity"
    assert packet.anomaly_score == MINING_SCORE
    assert packet.receipt_details["mode"] == MODE_A
    assert packet.public_interest_reason == (
        "USGS 3DEP bare-earth elevation reveals structural ground features in the "
        "Jacksonville mining district, Jackson County, Oregon; USGS MRDS records "
        "indicate a historic gold placer discovered in the 1850s."
    )


def test_mining_lead_seals_usgs_records_as_primary(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()))).packets
    sealed = {item.metadata["kind"]: item for item in packet.source_artifacts}
    assert set(sealed) == {"dem_export", "dem_catalog", "mrds_record"}
    assert {item.tier for item in sealed.values()} == {"tier_1_primary"}
    assert (
        sealed["dem_export"].sha256_hash
        == sealed["dem_export"].metadata["analysed_sha256"]
    )
    assert "dep_id=10042888" in sealed["mrds_record"].origin_url


def test_grid_that_changed_after_analysis_fails_the_evidence_bar(
    tmp_path: Path,
) -> None:
    mock = router(with_ditch(hillside()))
    served = [to_bsq(with_ditch(hillside())), to_bsq(with_ditch(hillside(seed=8)))]
    mock.routes[f"{ELEVATION_SERVICE}/exportImage"] = lambda _request: httpx.Response(
        200, content=served.pop(0), headers={"content-type": "application/octet-stream"}
    )
    result = harvest(tmp_path, mock)
    assert result.packets == [] and len(result.evidence_rejections) == 1


# --- Mode B and cultural-resource protection -----------------------------------------


def test_mound_without_a_mining_record_is_an_earthwork_lead(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_mound(flat()), sites=(FAR_GOLD,))).packets
    details = packet.receipt_details
    assert details["mode"] == MODE_B and packet.anomaly_score == 60.0
    assert details["documented"] is False
    assert packet.public_interest_reason == (
        "USGS 3DEP bare-earth elevation reveals structural ground features in Jackson "
        "County, Oregon; no published archaeological record lies within 500 m, and the "
        "precise location is withheld to protect cultural resources."
    )


COORDINATE_FRAGMENTS = ("42.3", "122.9", "-13689", "52091")


def public_text(packet: Any) -> str:
    return f"{packet.public_question} {packet.public_interest_reason}"


@pytest.mark.parametrize(
    "mock",
    [
        pytest.param(lambda: router(with_ditch(hillside())), id="mining"),
        pytest.param(lambda: router(with_mound(flat()), sites=()), id="earthwork"),
    ],
)
def test_public_text_never_gives_coordinates_or_record_ids(
    tmp_path: Path, mock: Any
) -> None:
    [packet] = harvest(tmp_path, mock()).packets
    exposed = f"{public_text(packet)} {packet.paste_ready_prompt}"
    for fragment in (*COORDINATE_FRAGMENTS, "10042888", "dep_id", "https://"):
        assert fragment not in exposed


def test_earthwork_region_is_the_county(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_mound(flat()), sites=())).packets
    assert packet.receipt_details["public_region"] == "Jackson County, Oregon"


def test_mining_region_without_a_district_falls_back_to_the_county(
    tmp_path: Path,
) -> None:
    record = mrds_record()
    del record["properties"]["districts"]
    [packet] = harvest(tmp_path, router(with_ditch(hillside()), record=record)).packets
    assert "features in Jackson County, Oregon;" in packet.public_interest_reason


def test_exact_location_stays_in_the_editor_only_receipt(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()))).packets
    details = packet.receipt_details
    editor = details["editor_only"]
    assert (editor["latitude"], editor["longitude"]) == (42.3115, -122.9725)
    south, west, north, east = editor["lidar_bounds"]["wgs84_south_west_north_east"]
    assert south < 42.3115 < north and west < -122.9725 < east
    assert 0.002 < north - south < 0.003  # 256 m of latitude
    assert len(editor["lidar_bounds"]["epsg_3857"]) == 4
    assert editor["mrds_record"]["dep_id"] == "10042888"
    [position] = editor["feature_positions"]
    assert position["kind"] == "contour_ditch"
    assert abs(position["latitude"] - 42.3115) < 0.001
    assert "Editor Eyes Only" in editor["handling"]
    # Nothing outside editor_only names the record or the position.
    outside = json.dumps({k: v for k, v in details.items() if k != "editor_only"})
    for fragment in ("10042888", "42.3115", "122.9725", "dep_id", "https://"):
        assert fragment not in outside


def test_every_sealed_source_is_restricted(tmp_path: Path) -> None:
    mock = router(
        with_mound(flat()), sites=(PLACER,), archive=open_context(oc_record())
    )
    [packet] = harvest(tmp_path, mock).packets
    assert packet.source_artifacts
    for item in packet.source_artifacts:
        assert item.metadata["restricted"] is True
        assert "Editor Eyes Only" in item.metadata["restricted_reason"]


def test_documented_site_is_named_and_sealed(tmp_path: Path) -> None:
    mock = router(with_mound(flat()), sites=(), archive=open_context(oc_record()))
    [packet] = harvest(tmp_path, mock).packets
    assert packet.receipt_details["documented"] is True
    assert (
        "Open Context publishes an archaeological record within 500 m"
        in packet.public_interest_reason
    )
    kinds_sealed = [item.metadata["kind"] for item in packet.source_artifacts]
    assert "open_context_record" in kinds_sealed


def test_area_near_human_remains_makes_no_lead(tmp_path: Path) -> None:
    mock = router(
        with_mound(flat()), sites=(), archive=open_context(oc_record(remains=True))
    )
    assert harvest(tmp_path, mock).packets == []
    assert mock.calls_to(f"{ELEVATION_SERVICE}/exportImage") == []


def test_no_lead_estimates_remaining_ore(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()))).packets
    text = (
        f"{packet.public_question} {packet.public_interest_reason} "
        f"{packet.paste_ready_prompt}"
    ).lower()
    for word in ("$", "/oz", "ounce", "value", "worth", "recover"):
        assert word not in text


# --- Five-Mask receipt and question ------------------------------------------------------


def test_public_question_names_the_area(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()))).packets
    assert packet.public_question == (
        "What historical mining infrastructure or ancient earthworks are documented "
        "at Jacksonville Placer Hill?"
    )
    question, reason = packet.public_question, packet.public_interest_reason
    assert public_interest(None, None, question, reason) == (question, reason)
    assert question.endswith("?") and reason.endswith(".")
    assert question == " ".join(question.split()) and reason == " ".join(reason.split())


def test_receipt_details_geometry_mineral_and_land(tmp_path: Path) -> None:
    [packet] = harvest(tmp_path, router(with_ditch(hillside()), land="USFS")).packets
    details = packet.receipt_details
    [ditch] = details["features"]
    assert set(ditch) == {
        "kind",
        "length_m",
        "width_m",
        "relief_m",
        "slope_gradient_divergence_deg",
        "fall_per_metre_along_feature",
    }
    assert ditch["kind"] == "contour_ditch" and ditch["length_m"] >= 150
    nearest = details["nearest_mineral_record"]
    assert (nearest["primary_commodity"], nearest["discovered"]) == ("gold", 1851)
    assert nearest["distance_km"] < 1.0
    assert details["public_land_status"] == "USFS"
    assert (
        details["lidar_raster"] == "OR_2022Jackson_C25" and details["lidar_cell_m"] == 1
    )
    assert "archaeologist" in details["cannot_show"]


@pytest.mark.parametrize(
    ("code", "status"),
    [("BLM", "BLM"), ("USFS", "USFS"), ("PVT", "private"), ("XYZ", "XYZ")],
)
def test_land_status(code: str, status: str) -> None:
    assert land_status([{"attributes": {"ADMIN_AGENCY_CODE": code}}]) == status
    assert land_status([]) == "unknown"


# --- Triage -------------------------------------------------------------------------------


def test_mining_lead_reaches_the_ledger(
    tmp_path: Path, db: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = ArchaeologyLidarAdapter(
        [AREA], **options(tmp_path, router(with_ditch(hillside())))
    )
    report = run_adapter(adapter, db, ollama()[0])
    assert report.rejected == [] and len(report.written) == 1
    row = db.execute(
        "SELECT anomaly_score, subject_domain, decomposition_receipt -> 'detector' -> 'mode' "
        "FROM triage_records WHERE adapter_name = 'trantor_antiquity'"
    ).fetchone()
    assert row == (70.0, "archaeological_antiquity", MODE_A)
    restricted = db.execute(
        """
        SELECT count(*) FROM restricted_sources r
        JOIN triage_records t ON r.source_id = ANY(t.source_ids)
        WHERE t.adapter_name = 'trantor_antiquity'
        """
    ).fetchone()
    assert restricted == (3,)

    again = ArchaeologyLidarAdapter(
        [AREA], **options(tmp_path, router(with_ditch(hillside())))
    )
    repeat = run_adapter(again, db, ollama()[0])
    assert repeat.written == [] and repeat.rejected == []


def test_earthwork_lead_passes_triage_on_usgs_records(tmp_path: Path, db: Conn) -> None:
    [packet] = harvest(tmp_path, router(with_mound(flat()), sites=())).packets
    report = triage_packets([packet], "archaeological_antiquity", db, ollama()[0], NOW)
    assert report.rejected == [] and report.written == [packet.lead_id]
