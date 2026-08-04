/**
 * Reads the physical slide-label photo out of a CZI, in the browser, without
 * uploading the file.
 *
 * Why this exists: the scanner records nothing about the staining -- HER2, DISH
 * and HE all carry an identical `TL Brightfield` channel and the same objective,
 * so the file's metadata cannot tell them apart and the doctor picks each file
 * by hand. The one place the stain *is* recorded is the label stuck on the glass,
 * which every CZI embeds as a `Label` attachment (HER2/DISH are machine-printed
 * with a DataMatrix, HE is handwritten). Showing it next to the picker lets the
 * doctor confirm the file they just chose is the slide they meant.
 *
 * Only byte ranges are read: the 120-byte file header, the attachment directory,
 * and the ~1.6 MB label blob -- never the multi-GB pixel data. File.slice() keeps
 * that to three short reads even on a 6 GB scan.
 *
 * Layout below is from the CZI spec and was verified byte-for-byte against
 * czifile's parse of the three slides in run-2026-07-29: the header's attachment
 * directory pointer, every directory entry, and the decoded label pixels all
 * match. The label sub-block is uncompressed Bgr24, which is why no image codec
 * is needed here.
 */

const ATTACHMENT_DIR_POSITION = 104   // int64 in the FileHeader segment
const DIRECTORY_ENTRY_HEADER = 32     // DirectoryEntryDV up to DimensionCount
const DIMENSION_ENTRY = 20            // one DimensionEntryDV1
const SUBBLOCK_FIXED = 16             // MetadataSize + AttachmentSize + DataSize
const SEGMENT_HEADER = 32             // id[16] + allocated_size + used_size
const ATTACHMENT_ENTRY = 128
const PIXEL_TYPE_BGR24 = 3

function magic(buf: ArrayBuffer, offset: number, length: number) {
  return new TextDecoder().decode(new Uint8Array(buf, offset, length))
}

/** A fixed-width name field, cut at its NUL padding. */
function cstring(buf: ArrayBuffer, offset: number, length: number) {
  const raw = magic(buf, offset, length)
  const end = raw.indexOf('\0')
  return end === -1 ? raw : raw.slice(0, end)
}

async function slice(file: File, start: number, end: number) {
  return file.slice(start, end).arrayBuffer()
}

/** Byte offset of the `Label` attachment segment, or null if the file has none. */
async function findLabelAttachment(file: File): Promise<number | null> {
  const header = await slice(file, 0, 120)
  if (!magic(header, 0, 10).startsWith('ZISRAWFILE')) return null
  const dirPos = Number(new DataView(header).getBigInt64(ATTACHMENT_DIR_POSITION, true))
  if (dirPos <= 0 || dirPos >= file.size) return null

  const dirHead = await slice(file, dirPos, dirPos + SEGMENT_HEADER + 4)
  if (!magic(dirHead, 0, 12).startsWith('ZISRAWATTDIR')) return null
  const count = new DataView(dirHead).getInt32(SEGMENT_HEADER, true)
  if (count <= 0 || count > 64) return null

  // Entries follow the segment header, the entry count and 252 bytes of padding.
  const entriesAt = dirPos + SEGMENT_HEADER + 256
  const entries = await slice(file, entriesAt, entriesAt + count * ATTACHMENT_ENTRY)
  const view = new DataView(entries)
  for (let i = 0; i < count; i++) {
    const at = i * ATTACHMENT_ENTRY
    // AttachmentEntryA1: SchemaType[2] Reserved[10] FilePosition ... Name[80]
    const name = cstring(entries, at + 48, 80)
    if (name === 'Label') return Number(view.getBigInt64(at + 12, true))
  }
  return null
}

/**
 * The label as a PNG data URL, rotated so the printed text reads horizontally
 * (the scanner stores it on its side), or null when the file carries no readable
 * label -- not a CZI, no Label attachment, or a compressed sub-block. A missing
 * preview is a silent non-event: it must never block picking the file.
 */
export async function readCziLabel(file: File): Promise<string | null> {
  try {
    const segmentAt = await findLabelAttachment(file)
    if (segmentAt === null) return null

    const segHead = await slice(file, segmentAt, segmentAt + SEGMENT_HEADER)
    const usedSize = Number(new DataView(segHead).getBigInt64(24, true))
    if (usedSize <= 256 || usedSize > 64 * 1024 * 1024) return null

    // The attachment's payload is itself a small CZI holding one sub-block.
    const blob = await slice(file, segmentAt + SEGMENT_HEADER + 256, segmentAt + SEGMENT_HEADER + usedSize)
    const bytes = new Uint8Array(blob)
    const at = indexOfAscii(bytes, 'ZISRAWSUBBLOCK')
    if (at < 0) return null

    const view = new DataView(blob)
    const metadataSize = view.getInt32(at + SEGMENT_HEADER, true)
    // DirectoryEntryDV: SchemaType[2] PixelType FilePosition FilePart Compression ...
    const entry = at + SEGMENT_HEADER + SUBBLOCK_FIXED
    const pixelType = view.getInt32(entry + 2, true)
    const compression = view.getInt32(entry + 18, true)
    if (pixelType !== PIXEL_TYPE_BGR24 || compression !== 0) return null

    const dimCount = view.getInt32(entry + 28, true)
    if (dimCount <= 0 || dimCount > 8) return null
    let width = 0
    let height = 0
    for (let i = 0; i < dimCount; i++) {
      const d = entry + DIRECTORY_ENTRY_HEADER + i * DIMENSION_ENTRY
      const dim = cstring(blob, d, 4)
      const size = view.getInt32(d + 8, true)
      if (dim === 'X') width = size
      if (dim === 'Y') height = size
    }
    if (!width || !height) return null

    // Fixed part + directory entry are padded to at least 256 bytes, then the
    // sub-block's own XML, then the pixels.
    const dirEntrySize = DIRECTORY_ENTRY_HEADER + dimCount * DIMENSION_ENTRY
    const pixelsAt = at + SEGMENT_HEADER + Math.max(256, SUBBLOCK_FIXED + dirEntrySize) + metadataSize
    if (pixelsAt + width * height * 3 > bytes.length) return null

    return toRotatedPng(bytes.subarray(pixelsAt), width, height)
  } catch {
    return null   // a malformed pick is the user's cue to choose another file
  }
}

function indexOfAscii(bytes: Uint8Array, needle: string) {
  const pattern = new TextEncoder().encode(needle)
  outer: for (let i = 0; i + pattern.length <= bytes.length; i++) {
    for (let j = 0; j < pattern.length; j++) if (bytes[i + j] !== pattern[j]) continue outer
    return i
  }
  return -1
}

/**
 * Bgr24 rows -> PNG data URL, turned a quarter turn clockwise so the printed
 * label text reads horizontally.
 *
 * The turn is done while filling the pixel buffer rather than with a canvas
 * transform, so that what the ImageData holds is exactly what gets drawn -- a
 * transform is invisible to a byte-level check, and this function is verified by
 * comparing its output against the same rotation computed independently.
 */
function toRotatedPng(bgr: Uint8Array, width: number, height: number): string | null {
  const outWidth = height
  const outHeight = width
  const rgba = new Uint8ClampedArray(outWidth * outHeight * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * 3
      const dst = (x * outWidth + (height - 1 - y)) * 4
      rgba[dst] = bgr[src + 2]
      rgba[dst + 1] = bgr[src + 1]
      rgba[dst + 2] = bgr[src]
      rgba[dst + 3] = 255
    }
  }
  const canvas = document.createElement('canvas')
  canvas.width = outWidth
  canvas.height = outHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.putImageData(new ImageData(rgba, outWidth, outHeight), 0, 0)
  return canvas.toDataURL('image/png')
}
