// Parses article Markdown into data, so that Astro escapes every piece of text.
// Only headings, paragraphs, and [S-XXXX] citations exist in ledger bodies.
import type { Article } from "./data";

export interface Segment {
  text: string;
  sourceId?: string;
}

export type Block =
  | { kind: "heading"; level: 2 | 3; text: string }
  | { kind: "paragraph"; segments: Segment[] };

const CITATION = /(\[S-\d+\])/;

function toSegments(text: string, known: Set<string>): Segment[] {
  return text
    .split(CITATION)
    .filter(Boolean)
    .map((piece) => {
      const sourceId = piece.slice(1, -1);
      return CITATION.test(piece) && known.has(sourceId)
        ? { text: piece, sourceId }
        : { text: piece };
    });
}

export function parseBody(article: Article): Block[] {
  const known = new Set(article.sources.map((item) => item.source_id));
  return article.body_markdown
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block): Block => {
      const heading = block.match(/^(#{1,3})\s+(.*)$/);
      if (heading?.[1] && heading[2]) {
        return {
          kind: "heading",
          level: heading[1].length === 3 ? 3 : 2,
          text: heading[2],
        };
      }
      return {
        kind: "paragraph",
        segments: toSegments(block.replace(/\s+/g, " "), known),
      };
    });
}

export function firstParagraph(article: Article): string {
  const block = parseBody(article).find((item) => item.kind === "paragraph");
  if (block?.kind !== "paragraph") {
    return "";
  }
  return block.segments
    .filter((item) => !item.sourceId)
    .map((item) => item.text)
    .join("")
    .trim();
}
