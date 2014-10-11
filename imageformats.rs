/*
    Copyright (c) 2014 Tero Hänninen

    MIT Software License

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#![allow(non_uppercase_statics)]    // for the static function pointers

extern crate flate;
use std::io::{File, BufferedReader, BufferedWriter, IoResult, IoError, OtherIoError};
use std::iter::{range_step};
use std::cmp::min;
use std::slice::bytes::{copy_memory};
use std::mem::{zeroed};
use self::flate::{inflate_bytes_zlib, deflate_bytes_zlib};

macro_rules! IFErr(
    ($e:expr) => (Err(IoError{kind: OtherIoError, desc: $e, detail: None}))
)

#[deriving(Show)]
pub struct IFImage {
    pub w      : uint,
    pub h      : uint,
    pub c      : ColFmt,
    pub pixels : Vec<u8>,
}

#[deriving(Show)]
pub struct IFInfo {
    pub w : uint,
    pub h : uint,
    pub c : ColFmt,
}

#[deriving(Show, Eq, PartialEq)]
pub enum ColFmt {
    FmtAuto = 0,
    FmtY = 1,
    FmtYA,
    FmtRGB,
    FmtRGBA,
    FmtBGR,
    FmtBGRA,
}

/** Returns: basic info about an image file. The color format information does
 * not correspond to the exact format in the file: for BGR/A data the format is
 * reported as RGB/A and for paletted images it might be RGB or RGBA or
 * whatever (paletted images are auto-depaletted by the decoders).
 */
#[allow(dead_code)]
pub fn read_image_info(filename: &str) -> IoResult<IFInfo> {
    let readfunc = match extract_extension(filename) {
        Some(".png")                 => read_png_info,
        Some(".tga")                 => read_tga_info,
        Some(".jpg") | Some(".jpeg") => read_jpeg_info,
        _ => return IFErr!("extension not recognized"),
    };
    let file = File::open(&Path::new(filename));
    let reader = &mut BufferedReader::new(file);
    readfunc(reader)
}

/** Paletted images are auto-depaletted.
 */
pub fn read_image(filename: &str, req_fmt: ColFmt) -> IoResult<IFImage> {
    let readfunc = match extract_extension(filename) {
        Some(".png")                 => read_png,
        Some(".tga")                 => read_tga,
        Some(".jpg") | Some(".jpeg") => read_jpeg,
        _ => return IFErr!("extension not recognized"),
    };
    let file = File::open(&Path::new(filename));
    let reader = &mut BufferedReader::new(file);
    readfunc(reader, req_fmt)
}

pub fn write_image(filename: &str, w: uint, h: uint, data: &[u8], tgt_fmt: ColFmt)
                                                                   -> IoResult<()>
{
    let writefunc = match extract_extension(filename) {
        Some(".png") => write_png,
        Some(".tga") => write_tga,
        _ => return IFErr!("extension not supported for writing"),
    };
    let file = File::create(&Path::new(filename));
    let writer = &mut BufferedWriter::new(file);
    writefunc(writer, w, h, data, tgt_fmt)
}

impl ColFmt {
    fn channels(&self) -> uint {
        match *self {
            FmtAuto           => 0,
            FmtY              => 1,
            FmtYA             => 2,
            FmtRGB  | FmtBGR  => 3,
            FmtRGBA | FmtBGRA => 4,
        }
    }
}

// ------------------------------------------------------------

#[deriving(Show)]
pub struct PngHeader {
    pub width              : u32,
    pub height             : u32,
    pub bit_depth          : u8,
    pub color_type         : u8,
    pub compression_method : u8,
    pub filter_method      : u8,
    pub interlace_method   : u8
}

static PNG_FILE_HEADER: [u8, ..8] =
    [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

pub fn read_png_info<R: Reader>(reader: &mut R) -> IoResult<IFInfo> {
    let hdr = try!(read_png_header(reader));

    let ctype: Option<PngColortype> = FromPrimitive::from_u8(hdr.color_type);
    let ctype = match ctype {
        Some(ct) => ct,
        None => return IFErr!("unsupported color type"),
    };
    let src_fmt = match ctype.color_channels() {
        1 => FmtY,
        2 => FmtYA,
        3 => FmtRGB,
        4 => FmtRGBA,
        _ => return IFErr!("internal error"),
    };

    Ok(IFInfo {
        w: hdr.width as uint,
        h: hdr.height as uint,
        c: src_fmt,
    })
}

pub fn read_png_header<R: Reader>(reader: &mut R) -> IoResult<PngHeader> {
    let mut buf = [0u8, ..33];  // file header + IHDR
    try!(reader.read_at_least(buf.len(), &mut buf));

    if !equal(buf[0..8], PNG_FILE_HEADER) ||
       !equal(buf[8..16], b"\0\0\0\x0dIHDR") ||
       !equal(buf[29..33], crc32be(buf[12..29]))
    {
        return IFErr!("corrupt png header");
    }

    Ok(PngHeader {
        width              : u32_from_be(buf[16..20]),
        height             : u32_from_be(buf[20..24]),
        bit_depth          : buf[24],
        color_type         : buf[25],
        compression_method : buf[26],
        filter_method      : buf[27],
        interlace_method   : buf[28],
    })
}

pub fn read_png<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    let req_fmt = match req_fmt {
        FmtAuto | FmtY | FmtYA | FmtRGB | FmtRGBA => req_fmt,
        _ => return IFErr!("format not supported")
    };

    let hdr = try!(read_png_header(reader));

    if hdr.width < 1 || hdr.height < 1 { return IFErr!("invalid dimensions") }
    if hdr.bit_depth != 8 { return IFErr!("only 8-bit images supported") }
    if hdr.compression_method != 0 || hdr.filter_method != 0 {
        return IFErr!("not supported");
    }

    let ilace: Option<PngInterlace> = FromPrimitive::from_u8(hdr.interlace_method);
    let ilace = match ilace {
        Some(im) => im,
        None => return IFErr!("unsupported interlace method"),
    };

    let ctype: Option<PngColortype> = FromPrimitive::from_u8(hdr.color_type);
    let ctype = match ctype {
        Some(ct) => ct,
        None => return IFErr!("unsupported color type"),
    };

    let src_fmt = match ctype.color_channels() {
        1 => FmtY,
        2 => FmtYA,
        3 => FmtRGB,
        4 => FmtRGBA,
        _ => return IFErr!("internal error"),
    };

    let dc = &mut PngDecoder {
        stream      : reader,
        w           : hdr.width as uint,
        h           : hdr.height as uint,
        ilace       : ilace,
        src_indexed : ctype == PngTypeIdx,
        src_fmt     : src_fmt,
        tgt_fmt     : if req_fmt == FmtAuto { src_fmt } else { req_fmt },
        chunkmeta   : unsafe { zeroed() },
        readbuf     : Vec::from_elem(4096, 0u8),
        uc_buf      : Vec::from_elem(0, 0u8),
        uc_start    : 0,
        crc         : Crc32::new(),
    };

    Ok(IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : try!(decode_png(dc))
    })
}

struct PngDecoder<'r, R:'r> {
    stream        : &'r mut R,
    w             : uint,
    h             : uint,
    ilace         : PngInterlace,
    src_indexed   : bool,
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,

    chunkmeta: [u8, ..12],   // for reading len, type, crc
    readbuf: Vec<u8>,
    uc_buf: Vec<u8>,
    uc_start: uint,
    crc: Crc32,
}

#[deriving(Eq, PartialEq)]
enum PngStage {
    IhdrParsed,
    PlteParsed,
    IdatParsed,
    //IendParsed,
}

fn decode_png<R: Reader>(dc: &mut PngDecoder<R>) -> IoResult<Vec<u8>> {

    let mut result = Vec::from_elem(0, 0u8);
    let mut stage = IhdrParsed;

    let mut palette = Vec::from_elem(0, 0u8);

    try!(dc.stream.read_at_least(8, dc.chunkmeta[mut 0..8]));
    loop {
        let mut len = u32_from_be(dc.chunkmeta[0..4]) as uint;
        if 0x7fff_ffff < len { return IFErr!("chunk too long"); }

        dc.crc.put(dc.chunkmeta[4..8]);   // type
        match dc.chunkmeta[4..8] {
            b"IDAT" => {
                if !(stage == IhdrParsed || (stage == PlteParsed && dc.src_indexed)) {
                    return IFErr!("corrupt chunk stream");
                }

                // also reads chunkmeta for next chunk
                result = try!(read_idat_stream(dc, len, &palette));
                stage = IdatParsed;
                continue;   // skip reading chunkmeta
            }
            b"PLTE" => {
                let entries = len / 3;
                if stage != IhdrParsed || len % 3 != 0 || 256 < entries {
                    return IFErr!("corrupt chunk stream");
                }
                palette = try!(dc.stream.read_exact(len));
                dc.crc.put(palette[]);
                try!(dc.stream.read_at_least(4, dc.chunkmeta[mut 0..4]));
                if !equal(dc.crc.finish_be(), dc.chunkmeta[0..4]) {
                    return IFErr!("corrupt chunk");
                }
                stage = PlteParsed;
            }
            b"IEND" => {
                if stage != IdatParsed {
                    return IFErr!("corrupt chunk stream");
                }
                try!(dc.stream.read_at_least(4, dc.chunkmeta[mut 0..4]));
                if len != 0 || !equal(dc.chunkmeta[0..4], [0xae, 0x42, 0x60, 0x82]) {
                    return IFErr!("corrupt chunk");
                }
                break;//stage = IendParsed;
            }
            _ => {
                // unknown chunk, ignore but check crc... or should crc be ignored?
                while 0 < len {
                    let bytes = min(len, dc.readbuf.len());
                    let got =
                        try!(dc.stream.read_at_least(bytes, dc.readbuf[mut 0..bytes]));
                    len -= got;
                    dc.crc.put(dc.readbuf[0..got]);
                }

                try!(dc.stream.read_at_least(4, dc.chunkmeta[mut 0..4]));
                if !equal(dc.crc.finish_be(), dc.chunkmeta[0..4]) {
                    return IFErr!("corrupt chunk");
                }
            }
        }

        try!(dc.stream.read_at_least(8, dc.chunkmeta[mut 0..8]));
    }

    Ok(result)
}

#[deriving(Eq, PartialEq, FromPrimitive)]
enum PngInterlace {
    PngIlaceNone, PngIlaceAdam7
}

#[deriving(Eq, PartialEq, FromPrimitive)]
enum PngColortype {
    PngTypeY    = 0,
    PngTypeRGB  = 2,
    PngTypeIdx  = 3,
    PngTypeYA   = 4,
    PngTypeRGBA = 6,
}

impl PngColortype {
    fn color_channels(&self) -> i64 {
        match *self {
            PngTypeY                => 1,
            PngTypeRGB | PngTypeIdx => 3,
            PngTypeYA               => 2,
            PngTypeRGBA             => 4,
        }
    }
}

fn read_idat_stream<R: Reader>(dc: &mut PngDecoder<R>, mut len: uint, palette: &Vec<u8>)
                                                                    -> IoResult<Vec<u8>>
{
    let filter_step = if dc.src_indexed { 1 } else { dc.src_fmt.channels() };
    let tgt_bytespp = dc.tgt_fmt.channels() as uint;
    let tgt_linesize = dc.w as uint * tgt_bytespp;

    let mut result = Vec::from_elem(dc.w as uint * dc.h as uint * tgt_bytespp, 0u8);
    let mut depaletted_line = if dc.src_indexed {
        Vec::from_elem((dc.w * 3) as uint, 0u8)
    } else {
        Vec::from_elem(0, 0u8)
    };

    let chan_convert = try!(get_converter(dc.src_fmt, dc.tgt_fmt));

    let depalette_convert = |src_line: &[u8], tgt_line: &mut[u8]| -> IoResult<()> {
        let mut d = 0u;
        for s in range(0, src_line.len()) {
            let pidx = src_line[s] as uint * 3;
            if palette.len() < pidx + 3 {
                return IFErr!("palette index invalid");
            }
            copy_memory(depaletted_line[mut d..d+3], palette[pidx..pidx+3]);
            d += 3;
        }
        Ok(chan_convert(depaletted_line[0 .. src_line.len()*3], tgt_line))
    };

    let simple_convert = |src_line: &[u8], tgt_line: &mut[u8]| -> IoResult<()> {
        Ok(chan_convert(src_line, tgt_line))
    };

    let convert = if dc.src_indexed { depalette_convert } else { simple_convert };

    try!(fill_uc_buf(dc, &mut len));

    if dc.ilace == PngIlaceNone {
        let src_linesize = dc.w * filter_step;
        let mut cline = Vec::from_elem(src_linesize+1, 0u8);  // current line + filter byte
        let mut pline = Vec::from_elem(src_linesize+1, 0u8);  // previous line

        let mut ti = 0u;
        for _j in range(0, dc.h) {
            next_uncompressed_line(dc, cline[mut]);
            let filter_type: u8 = cline[0];

            try!(recon(
                cline[mut 1 .. src_linesize+1], pline[mut 1 .. src_linesize+1],
                filter_type, filter_step
            ));
            try!(convert(cline[1 .. cline.len()], result[mut ti .. ti+tgt_linesize]));

            ti += tgt_linesize;

            let swap = pline;
            pline = cline;
            cline = swap;
        }
    } else {
        // Adam7 interlacing

        let redw: [uint, ..7] = [
            (dc.w + 7) / 8,
            (dc.w + 3) / 8,
            (dc.w + 3) / 4,
            (dc.w + 1) / 4,
            (dc.w + 1) / 2,
            (dc.w + 0) / 2,
            (dc.w + 0) / 1,
        ];
        let redh: [uint, ..7] = [
            (dc.h + 7) / 8,
            (dc.h + 7) / 8,
            (dc.h + 3) / 8,
            (dc.h + 3) / 4,
            (dc.h + 1) / 4,
            (dc.h + 1) / 2,
            (dc.h + 0) / 2,
        ];

        let max_scanline_size = dc.w * filter_step;
        let mut linebuf0 = Vec::from_elem(max_scanline_size+1, 0u8); // +1 for filter type
        let mut linebuf1 = Vec::from_elem(max_scanline_size+1, 0u8); // +1 for filter type
        let mut redlinebuf = Vec::from_elem(dc.w * tgt_bytespp, 0u8);

        for pass in range(0, 7) {
            let tgt_px: A7IdxTranslator = A7_IDX_TRANSLATORS[pass];   // target pixel
            let src_linesize = redw[pass] * filter_step;
            let mut cline = linebuf0[mut 0 .. src_linesize+1];
            let mut pline = linebuf1[mut 0 .. src_linesize+1];

            for j in range(0, redh[pass]) {
                next_uncompressed_line(dc, cline[mut]);
                let filter_type: u8 = cline[0];

                try!(recon(
                    cline[mut 1 .. src_linesize+1], pline.slice(1, src_linesize+1),
                    filter_type, filter_step
                ));
                try!(convert(
                    cline.slice(1, cline.len()),
                    redlinebuf[mut 0..redw[pass] * tgt_bytespp]
                ));

                let mut redi = 0u;
                for i in range(0, redw[pass]) {
                    let tgt = tgt_px(i, j, dc.w) * tgt_bytespp;
                    copy_memory(
                        result[mut tgt .. tgt+tgt_bytespp],
                        redlinebuf[redi .. redi+tgt_bytespp]
                    );
                    redi += tgt_bytespp;
                }

                let swap = pline;
                pline = cline;
                cline = swap;
            }
        }
    }

    return Ok(result);
}

type A7IdxTranslator = fn(redx: uint, redy: uint, dstw: uint) -> uint;
static A7_IDX_TRANSLATORS: [A7IdxTranslator, ..7] = [
    a7_red1_to_dst,
    a7_red2_to_dst,
    a7_red3_to_dst,
    a7_red4_to_dst,
    a7_red5_to_dst,
    a7_red6_to_dst,
    a7_red7_to_dst,
];

fn a7_red1_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { redy*8*dstw + redx*8     }
fn a7_red2_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { redy*8*dstw + redx*8+4   }
fn a7_red3_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { (redy*8+4)*dstw + redx*4 }
fn a7_red4_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { redy*4*dstw + redx*4+2   }
fn a7_red5_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { (redy*4+2)*dstw + redx*2 }
fn a7_red6_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { redy*2*dstw + redx*2+1   }
fn a7_red7_to_dst(redx:uint, redy:uint, dstw:uint) -> uint { (redy*2+1)*dstw + redx   }

fn fill_uc_buf<R: Reader>(dc: &mut PngDecoder<R>, len: &mut uint) -> IoResult<()> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut totallen = 0u;
    loop {
        let mut fresh = Vec::from_elem(*len, 0u8);
        try!(dc.stream.read_at_least(*len, fresh[mut]));
        dc.crc.put(fresh[]);
        chunks.push(fresh);
        totallen += *len;

        // crc
        try!(dc.stream.read_at_least(4, dc.chunkmeta[mut 0..4]));
        if !equal(dc.crc.finish_be(), dc.chunkmeta[0..4]) {
            return IFErr!("corrupt image data");
        }

        // next chunk's len and type
        try!(dc.stream.read_at_least(8, dc.chunkmeta[mut 0..8]));
        *len = u32_from_be(dc.chunkmeta[0..4]) as uint;
        if dc.chunkmeta[4..8] != b"IDAT" {
            break;
        }
    }

    let mut alldata = Vec::from_elem(totallen, 0u8);
    let mut di = 0u;
    for chunk in chunks.iter() {
        copy_memory(alldata[mut di .. di+chunk.len()], chunk[]);
        di += chunk.len();
    }

    let inflated = match inflate_bytes_zlib(alldata[]) {
        Some(cvec) => cvec,
        None => return IFErr!("could not inflate zlib source")
    };

    dc.uc_buf = Vec::from_elem(inflated.as_slice().len(), 0u8);
    copy_memory(dc.uc_buf[mut], inflated.as_slice());

    Ok(())
}

fn next_uncompressed_line<R: Reader>(dc: &mut PngDecoder<R>, dst: &mut[u8]) {
    let dstlen = dst.len();
    copy_memory(dst, dc.uc_buf[dc.uc_start .. dc.uc_start + dstlen]);
    dc.uc_start += dst.len();
}

fn recon(cline: &mut[u8], pline: &[u8], ftype: u8, fstep: uint) -> IoResult<()> {
    let ftype: Option<PngFilterType> = FromPrimitive::from_u8(ftype);
    match ftype {
        Some(PngFilterNone)
            => { }
        Some(PngFilterSub) => {
            for k in range(fstep, cline.len()) {
                cline[k] += cline[k-fstep];
            }
        }
        Some(PngFilterUp) => {
            for k in range(0, cline.len()) {
                cline[k] += pline[k];
            }
        }
        Some(PngFilterAverage) => {
            for k in range(0, fstep) {
                cline[k] += pline[k] / 2;
            }
            for k in range(fstep, cline.len()) {
                cline[k] +=
                    ((cline[k-fstep] as uint + pline[k] as uint) / 2) as u8;
            }
        }
        Some(PngFilterPaeth) => {
            for k in range(0, fstep) {
                cline[k] += paeth(0, pline[k], 0);
            }
            for k in range(fstep, cline.len()) {
                cline[k] += paeth(cline[k-fstep], pline[k], pline[k-fstep]);
            }
        }
        None => return IFErr!("filter type not supported"),
    }
    Ok(())
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let mut pc = c as int;
    let mut pa = b as int - pc;
    let mut pb = a as int - pc;
    pc = pa + pb;
    if pa < 0 { pa = -pa; }
    if pb < 0 { pb = -pb; }
    if pc < 0 { pc = -pc; }

    if pa <= pb && pa <= pc {
        return a;
    } else if pb <= pc {
        return b;
    }
    return c;
}

#[deriving(FromPrimitive)]
enum PngFilterType {
    PngFilterNone = 0,
    PngFilterSub,
    PngFilterUp,
    PngFilterAverage,
    PngFilterPaeth,
}

// --------------------------------------------------
// PNG encoder

pub fn write_png<W: Writer>(writer: &mut W, w: uint, h: uint, data: &[u8], tgt_fmt: ColFmt)
                                                                            -> IoResult<()>
{
    let src_chans = data.len() / w / h;
    if w < 1 || h < 1 || (src_chans * w * h != data.len()) {
        return IFErr!("invalid dimensions or data length");
    }

    let src_fmt = match src_chans {
        1 => FmtY,
        2 => FmtYA,
        3 => FmtRGB,
        4 => FmtRGBA,
        _ => return IFErr!("format not supported"),
    };

    let tgt_fmt = match tgt_fmt {
        FmtAuto                         => src_fmt,
        FmtY | FmtYA | FmtRGB | FmtRGBA => tgt_fmt,
        _ => return IFErr!("invalid format"),
    };

    let ec = &mut PngEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_chans : src_chans,
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        data      : data,
        crc       : Crc32::new(),
    };

    try!(write_png_header(ec));
    try!(write_png_image_data(ec));

    let iend: &'static[u8] = b"\0\0\0\0IEND\xae\x42\x60\x82";
    ec.stream.write(iend)
}

fn write_png_header<W: Writer>(ec: &mut PngEncoder<W>) -> IoResult<()> {
    let mut hdr: [u8, ..33] = unsafe { zeroed() };

    copy_memory(hdr[mut 0..8], PNG_FILE_HEADER);
    copy_memory(hdr[mut 8..16], b"\0\0\0\x0dIHDR");
    copy_memory(hdr[mut 16..20], u32_to_be(ec.w as u32));
    copy_memory(hdr[mut 20..24], u32_to_be(ec.h as u32));
    hdr[24] = 8;    // bit depth
    hdr[25] = match ec.tgt_fmt {    // color type
        FmtY => PngTypeY,
        FmtYA => PngTypeYA,
        FmtRGB => PngTypeRGB,
        FmtRGBA => PngTypeRGBA,
        _ => return IFErr!("not supported"),
    } as u8;
    copy_memory(hdr[mut 26..29], [0, 0, 0]);  // compression, filter, interlace
    ec.crc.put(hdr[12..29]);
    copy_memory(hdr[mut 29..33], ec.crc.finish_be());

    ec.stream.write(hdr)
}

struct PngEncoder<'r, W:'r> {
    stream        : &'r mut W,   // TODO is this ok?
    w             : uint,
    h             : uint,
    src_chans     : uint,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    data          : &'r [u8],
    crc           : Crc32,
}

fn write_png_image_data<W: Writer>(ec: &mut PngEncoder<W>) -> IoResult<()> {
    let convert = try!(get_converter(ec.src_fmt, ec.tgt_fmt));

    let filter_step = ec.tgt_fmt.channels();
    let tgt_linesize = ec.w * filter_step + 1;   // +1 for filter type
    let mut cline = Vec::from_elem(tgt_linesize, 0u8);
    let mut pline = Vec::from_elem(tgt_linesize, 0u8);
    let mut filtered_image = Vec::from_elem(tgt_linesize * ec.h, 0u8);

    let src_linesize = ec.w * ec.src_chans;

    let mut ti = 0u;
    for si in range_step(0, ec.h * src_linesize, src_linesize) {
        convert(ec.data[si .. si+src_linesize], cline[mut 1 .. tgt_linesize]);

        for i in range(1, filter_step+1) {
            filtered_image[mut][ti+i] = cline[i] - paeth(0, pline[i], 0)
        }
        for i in range(filter_step+1, cline.len()) {
            filtered_image[mut][ti+i] =
                cline[i] - paeth(cline[i-filter_step], pline[i], pline[i-filter_step])
        }

        filtered_image[mut][ti] = PngFilterPaeth as u8;

        let swap = pline;
        pline = cline;
        cline = swap;

        ti += tgt_linesize;
    }

    let compressed = match deflate_bytes_zlib(filtered_image[]) {
        Some(cvec) => cvec,
        None => return IFErr!("compression failed"),
    };
    ec.crc.put(b"IDAT");
    ec.crc.put(compressed.as_slice());
    let crc = &ec.crc.finish_be();

    // TODO split up data into smaller chunks?
    let chunklen = compressed.as_slice().len() as u32;
    try!(ec.stream.write(u32_to_be(chunklen)));
    try!(ec.stream.write(b"IDAT"));
    try!(ec.stream.write(compressed.as_slice()));
    try!(ec.stream.write(crc));
    Ok(())
}

// ------------------------------------------------------------

#[deriving(Show)]
pub struct TgaHeader {
   pub id_length      : u8,
   pub palette_type   : u8,
   pub data_type      : u8,
   pub palette_start  : u16,
   pub palette_length : u16,
   pub palette_bits   : u8,
   pub x_origin       : u16,
   pub y_origin       : u16,
   pub width          : u16,
   pub height         : u16,
   pub bits_pp        : u8,
   pub flags          : u8,
}

/** Returns: basic info. NOTE: Even if the file contains BGR/A data it's reported as
 * RGB/A. Returns error if data type is not supported by the decoder, at least
 * for now (don't rely on it, it's not that way on purpose).
 */
pub fn read_tga_info<R: Reader>(reader: &mut R) -> IoResult<IFInfo> {
    let hdr = try!(read_tga_header(reader));
    let TgaInfo { src_chans, src_fmt, rle } = try!(parse_tga_header(&hdr));
    let _src_chans = src_chans; let _rle = rle; // warnings be gone

    let reported_fmt = match src_fmt {
        FmtY => FmtY,
        FmtYA => FmtYA,
        FmtBGR => FmtRGB,
        FmtBGRA => FmtRGBA,
        _ => return IFErr!("source format unknown"),
    };

    Ok(IFInfo {
        w: hdr.width as uint,
        h: hdr.height as uint,
        c: reported_fmt,
    })
}

pub fn read_tga_header<R: Reader>(reader: &mut R) -> IoResult<TgaHeader> {
    let mut buf = [0u8, ..18];
    try!(reader.read_at_least(buf.len(), &mut buf));

    Ok(TgaHeader {
        id_length      : buf[0],
        palette_type   : buf[1],
        data_type      : buf[2],
        palette_start  : u16_from_le(buf[3..5]),
        palette_length : u16_from_le(buf[5..7]),
        palette_bits   : buf[7],
        x_origin       : u16_from_le(buf[8..10]),
        y_origin       : u16_from_le(buf[10..12]),
        width          : u16_from_le(buf[12..14]),
        height         : u16_from_le(buf[14..16]),
        bits_pp        : buf[16],
        flags          : buf[17],
    })
}

pub fn read_tga<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    let hdr = try!(read_tga_header(reader));

    if 0 < hdr.palette_type { return IFErr!("paletted TGAs not supported"); }
    if hdr.width < 1 || hdr.height < 1 { return IFErr!("invalid dimensions"); }
    if 0 < (hdr.flags & 0xc0) { return IFErr!("interlaced TGAs not supported"); }
    if 0 < (hdr.flags & 0x10) { return IFErr!("right-to-left TGAs not supported"); }

    let TgaInfo { src_chans, src_fmt, rle } = try!(parse_tga_header(&hdr));

    let tgt_fmt = match req_fmt {
        FmtY | FmtYA | FmtRGB | FmtRGBA => req_fmt,
        FmtAuto => match src_fmt {
            FmtY => FmtY,
            FmtYA => FmtYA,
            FmtBGR => FmtRGB,
            FmtBGRA => FmtRGBA,
            _ => return IFErr!("not supported"),
        },
        _ => return IFErr!("conversion not supported"),
    };

    try!(skip(reader, hdr.id_length as uint));

    let dc = &mut TgaDecoder {
        stream         : reader,
        w              : hdr.width as uint,
        h              : hdr.height as uint,
        origin_at_top  : 0 < (hdr.flags & 0x20),
        src_chans      : src_chans,
        rle            : rle,
        src_fmt        : src_fmt,
        tgt_fmt        : tgt_fmt,
    };

    Ok(IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : try!(decode_tga(dc))
    })
}

struct TgaInfo {
    src_chans : uint,
    src_fmt   : ColFmt,
    rle       : bool,
}

// Returns: source color format and whether it's RLE compressed
fn parse_tga_header(hdr: &TgaHeader) -> IoResult<TgaInfo> {
    let mut rle = false;
    let datatype: Option<TgaDataType> = FromPrimitive::from_u8(hdr.data_type);
    let attr_bits_pp = hdr.flags & 0xf;
    match (datatype, hdr.bits_pp, attr_bits_pp) {
        (Some(TrueColor),    24, 0) => {}
        (Some(TrueColor),    32, 8) => {}
        (Some(TrueColor),    32, 0) => {} // some pics say 0 bits for attr although 8 bits present
        (Some(Gray),          8, 0) => {}
        (Some(Gray),         16, 8) => {}
        (Some(TrueColorRLE), 24, 0) => { rle = true; }
        (Some(TrueColorRLE), 32, 8) => { rle = true; }
        (Some(TrueColorRLE), 32, 0) => { rle = true; }  // again, 8 bits are present in some pics
        (Some(GrayRLE),       8, 0) => { rle = true; }
        (Some(GrayRLE),      16, 8) => { rle = true; }
        _ => return IFErr!("data type not supported")
    }

    let src_chans = hdr.bits_pp as uint / 8;
    let src_fmt = match src_chans {
        1 => FmtY,
        2 => FmtYA,
        3 => FmtBGR,
        4 => FmtBGRA,
        _ => return IFErr!("not supported"),
    };

    Ok(TgaInfo {
        src_chans : src_chans,
        src_fmt   : src_fmt,
        rle       : rle,
    })
}

fn decode_tga<R: Reader>(dc: &mut TgaDecoder<R>) -> IoResult<Vec<u8>> {
    let tgt_chans = dc.tgt_fmt.channels();
    let tgt_linesize = (dc.w * tgt_chans) as i64;
    let src_linesize = (dc.w * dc.src_chans) as uint;

    let mut src_line = Vec::from_elem(src_linesize, 0u8);
    let mut result = Vec::from_elem((dc.w * dc.h * tgt_chans) as uint, 0u8);

    let (tgt_stride, mut ti): (i64, i64) =
        if dc.origin_at_top {
            (tgt_linesize, 0)
        } else {
            (-tgt_linesize, (dc.h-1) as i64 * tgt_linesize)
        };

    let convert: LineConverter = try!(get_converter(dc.src_fmt, dc.tgt_fmt));

    if !dc.rle {
        for _j in range(0, dc.h) {
            try!(dc.stream.read_at_least(src_linesize, src_line[mut]));
            convert(src_line[], result[mut ti as uint..(ti+tgt_linesize) as uint]);
            ti += tgt_stride;
        }
        return Ok(result);
    }

    // ---- RLE ----

    let bytes_pp = dc.src_chans as uint;
    let mut rbuf = Vec::from_elem(src_linesize, 0u8);
    let mut plen = 0u;    // packet length
    let mut its_rle = false;

    for _ in range(0, dc.h) {
        // fill src_line with uncompressed data
        let mut wanted: uint = src_linesize;
        while 0 < wanted {
            if plen == 0 {
                let hdr = try!(dc.stream.read_u8()) as uint;
                its_rle = 0 < (hdr & 0x80);
                plen = ((hdr & 0x7f) + 1) * bytes_pp;
            }
            let gotten: uint = src_linesize - wanted;
            let copysize: uint = min(plen, wanted);
            if its_rle {
                try!(dc.stream.read_at_least(bytes_pp, rbuf[mut 0..bytes_pp]));
                for p in range_step(gotten, gotten+copysize, bytes_pp) {
                    copy_memory(src_line[mut p..p+bytes_pp], rbuf[0..bytes_pp]);
                }
            } else {    // it's raw
                let slice: &mut[u8] = src_line[mut gotten..gotten+copysize];
                try!(dc.stream.read_at_least(copysize, slice));
            }
            wanted -= copysize;
            plen -= copysize;
        }

        convert(src_line[], result[mut ti as uint .. (ti+tgt_linesize) as uint]);
        ti += tgt_stride;
    }

    Ok(result)
}

struct TgaDecoder<'r, R:'r> {
    stream        : &'r mut R,   // TODO is this ok?
    w             : uint,
    h             : uint,
    origin_at_top : bool,    // src
    src_chans     : uint,
    rle           : bool,          // run length compressed
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,
}

#[deriving(FromPrimitive)]
enum TgaDataType {
    Idx          = 1,
    TrueColor    = 2,
    Gray         = 3,
    IdxRLE       = 9,
    TrueColorRLE = 10,
    GrayRLE      = 11,
}

// --------------------------------------------------
// TGA Encoder

/// For tgt_fmt, accepts FmtRGB/A but not FmtBGR/A; will encode as BGR/A.
pub fn write_tga<W: Writer>(writer: &mut W, w: uint, h: uint, data: &[u8],
                                         tgt_fmt: ColFmt) -> IoResult<()> {
    if w < 1 || h < 1 || 0xffff < w || 0xffff < h {
        return IFErr!("invalid dimensions");
    }

    let src_chans = data.len() / w / h;
    if src_chans * w * h != data.len() {
        return IFErr!("mismatching dimensions and length");
    }

    let src_fmt = match src_chans {
        1 => FmtY,
        2 => FmtYA,
        3 => FmtRGB,
        4 => FmtRGBA,
        _ => return IFErr!("format not supported"),
    };

    let tgt_fmt = match (src_fmt, tgt_fmt) {
        (FmtY, FmtAuto) | (FmtYA, FmtAuto) => src_fmt,
        (FmtRGB, FmtAuto)                  => FmtBGR,
        (FmtRGBA, FmtAuto)                 => FmtBGRA,
        (_, FmtY) | (_, FmtYA)             => tgt_fmt,
        (_, FmtRGB)                        => FmtBGR,
        (_, FmtRGBA)                       => FmtBGRA,
        _ => return IFErr!("invalid format"),
    };

    let ec = &mut TgaEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_chans : src_chans,
        tgt_chans : tgt_fmt.channels(),
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        rle       : true,
        data      : data,
    };

    try!(write_tga_header(ec));
    try!(write_tga_image_data(ec));

    // write footer
    let ftr: &'static [u8] =
        b"\x00\x00\x00\x00\
          \x00\x00\x00\x00\
          TRUEVISION-XFILE.\x00";
    try!(ec.stream.write(ftr));

    ec.stream.flush()
}

fn write_tga_header<W: Writer>(ec: &mut TgaEncoder<W>) -> IoResult<()> {
    let (data_type, has_alpha) = match ec.tgt_chans {
        1 => (if ec.rle { GrayRLE      } else { Gray      }, false),
        2 => (if ec.rle { GrayRLE      } else { Gray      }, true),
        3 => (if ec.rle { TrueColorRLE } else { TrueColor }, false),
        4 => (if ec.rle { TrueColorRLE } else { TrueColor }, true),
        _ => return IFErr!("internal error")
    };

    let w = u16_to_le(ec.w as u16);
    let h = u16_to_le(ec.h as u16);
    let hdr: &[u8, ..18] = &[
        0, 0,
        data_type as u8,
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
        w[0], w[1],
        h[0], h[1],
        (ec.tgt_chans * 8) as u8,
        if has_alpha {8u8} else {0u8},  // flags
    ];

    ec.stream.write(hdr)
}

fn write_tga_image_data<W: Writer>(ec: &mut TgaEncoder<W>) -> IoResult<()> {
    let src_linesize = (ec.w * ec.src_chans) as uint;
    let tgt_linesize = (ec.w * ec.tgt_chans) as uint;
    let mut tgt_line = Vec::from_elem(tgt_linesize, 0u8);
    let mut si = (ec.h-1) as uint * src_linesize;

    let convert = try!(get_converter(ec.src_fmt, ec.tgt_fmt));

    if !ec.rle {
        for _ in range(0, ec.h) {
            convert(ec.data[si..si+src_linesize], tgt_line[mut]);
            try!(ec.stream.write(tgt_line[]));
            si -= src_linesize; // origin at bottom
        }
        return Ok(());
    }

    // ----- RLE -----

    let bytes_pp = ec.tgt_chans as uint;
    let max_packets_per_line = (tgt_linesize+127) / 128;
    let mut cmp_buf = Vec::from_elem(tgt_linesize+max_packets_per_line, 0u8);
    for _ in range(0, ec.h) {
        convert(ec.data[si..si+src_linesize], tgt_line[mut]);
        let compressed_line = rle_compress(tgt_line[], cmp_buf[mut], ec.w, bytes_pp);
        try!(ec.stream.write(compressed_line[]));
        si -= src_linesize;
    }
    return Ok(());
}

fn rle_compress<'a>(line: &[u8], cmp_buf: &'a mut[u8], w: uint, bytes_pp: uint)
                                                                    -> &'a [u8]
{
    let rle_limit = if 1 < bytes_pp { 2 } else { 3 };   // run len worth an RLE packet
    let mut rawlen = 0u;
    let mut raw_i = 0u;   // start of raw packet data
    let mut cmp_i = 0u;
    let mut pixels_left = w;
    let mut px: &[u8];

    let mut i = bytes_pp;
    while 0 < pixels_left {
        let mut runlen = 1u;
        px = line[i-bytes_pp .. i];
        while i < line.len() && equal(px, line[i..i+bytes_pp]) && runlen < 128 {
            runlen += 1;
            i += bytes_pp;
        }
        pixels_left -= runlen;

        if runlen < rle_limit {
            // data goes to raw packet
            rawlen += runlen;
            if 128 <= rawlen {  // full packet, need to store it
                let copysize = 128 * bytes_pp;
                cmp_buf[cmp_i] = 0x7f; cmp_i += 1;  // packet header
                copy_memory(
                    cmp_buf[mut cmp_i..cmp_i+copysize],
                    line[raw_i..raw_i+copysize]
                );
                cmp_i += copysize;
                raw_i += copysize;
                rawlen -= 128;
            }
        } else {
            // RLE packet is worth it
            // store raw packet first, when needed
            if 0 < rawlen {
                let copysize = rawlen * bytes_pp;
                cmp_buf[cmp_i] = (rawlen-1) as u8;    // packet header
                cmp_i += 1;
                copy_memory(
                    cmp_buf[mut cmp_i..cmp_i+copysize],
                    line[raw_i..raw_i+copysize]
                );
                cmp_i += copysize;
                rawlen = 0;
            }

            // store RLE packet
            cmp_buf[cmp_i] = (0x80 | (runlen-1)) as u8;   // packet header
            cmp_i += 1;
            copy_memory(
                cmp_buf[mut cmp_i..cmp_i+bytes_pp],
                px[0..bytes_pp]
            );
            cmp_i += bytes_pp;
            raw_i = i;
        }
        i += bytes_pp;
    }   // for

    if 0 < rawlen {     // last packet of the line, if any
        let copysize = rawlen * bytes_pp;
        cmp_buf[cmp_i] = (rawlen-1) as u8;    // packet header
        cmp_i += 1;
        copy_memory(
            cmp_buf[mut cmp_i..cmp_i+copysize],
            line[raw_i..raw_i+copysize]
        );
        cmp_i += copysize;
    }

    cmp_buf.slice(0, cmp_i)
}

struct TgaEncoder<'r, R:'r> {
    stream        : &'r mut R,   // TODO is this ok?
    w             : uint,
    h             : uint,
    src_chans     : uint,
    tgt_chans     : uint,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    rle           : bool,          // run length compressed
    data          : &'r [u8],
}

// ------------------------------------------------------------
// Baseline JPEG decoder

pub fn read_jpeg_info<R: Reader>(stream: &mut R) -> IoResult<IFInfo> {
    try!(read_jfif(stream));

    loop {
        let mut marker: [u8, ..2] = [0, 0];
        try!(stream.read_at_least(marker.len(), marker));

        if marker[0] != 0xff { return IFErr!("no marker"); }
        while marker[1] == 0xff {
            try!(stream.read_at_least(1, marker[mut 1..2]));
        }

        match marker[1] {
            SOF0 | SOF2 => {
                let mut tmp: [u8, ..8] = [0,0,0,0, 0,0,0,0];
                try!(stream.read_at_least(tmp.len(), tmp));
                return Ok(IFInfo {
                    w: u16_from_be(tmp[5..7]) as uint,
                    h: u16_from_be(tmp[3..5]) as uint,
                    c: match tmp[7] {
                           1 => FmtY,
                           3 => FmtRGB,
                           _ => return IFErr!("not valid baseline jpeg")
                       },
                });
            }
            SOS | EOI => return IFErr!("no frame header"),
            DRI | DHT | DQT | COM | APP0 ... APPF => {
                let mut tmp: [u8, ..2] = [0, 0];
                try!(stream.read_at_least(tmp.len(), tmp));
                let len = u16_from_be(tmp) - 2;
                try!(skip(stream, len as uint));
            }
            _ => return IFErr!("invalid / unsupported marker"),
        }
    }
}

pub fn read_jpeg<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    let req_fmt = match req_fmt {
        FmtAuto | FmtY | FmtYA | FmtRGB | FmtRGBA => req_fmt,
        _ => return IFErr!("format not supported")
    };

    try!(read_jfif(reader));

    let dc = &mut JpegDecoder {
        stream      : reader,
        w           : 0,
        h           : 0,
        tgt_fmt     : req_fmt,
        eoi_reached : false,
        has_frame_header : false,
        qtables     : unsafe { zeroed() },
        ac_tables   : unsafe { zeroed() },
        dc_tables   : unsafe { zeroed() },
        cb          : 0,
        bits_left   : 0,
        num_mcu_x   : 0,
        num_mcu_y   : 0,
        restart_interval : 0,
        comps       : unsafe { zeroed() },
        index_for   : [0, 0, 0],
        num_comps   : 0,
        hmax        : 0,
        vmax        : 0,
    };

    try!(read_markers(dc));   // reads until first scan header

    for c in range(0, dc.comps.len()) {
        dc.comps[c].data = Vec::from_elem(0, 0u8);
    }

    if dc.eoi_reached {
        return IFErr!("no image data");
    }
    dc.tgt_fmt =
        if req_fmt == FmtAuto {
            match dc.num_comps {
                1 => FmtY, 3 => FmtRGB,
                _ => return IFErr!("internal error")
            }
        } else {
            req_fmt
        };

    Ok(IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : try!(decode_jpeg(dc))
    })
}

fn read_jfif<R: Reader>(reader: &mut R) -> IoResult<()> {
    let mut buf: [u8, ..20] = unsafe { zeroed() }; // SOI, APP0
    try!(reader.read_at_least(buf.len(), buf));

    let len = u16_from_be(buf[4..6]) as uint;

    if !equal(buf[0..4], [0xff_u8, 0xd8, 0xff, 0xe0]) ||
       !equal(buf[6..11], b"JFIF\0") || len < 16 {
        return IFErr!("not JPEG/JFIF");
    }

    if buf[11] != 1 {
        return IFErr!("version not supported");
    }

    // ignore density_unit, -x, -y at 13, 14..16, 16..18

    let thumbsize = buf[18] as uint * buf[19] as uint * 3;
    if thumbsize != len - 16 {
        return IFErr!("corrupt jpeg header");
    }
    skip(reader, thumbsize)
}

struct JpegDecoder<'r, R:'r> {
    stream        : &'r mut R,   // TODO is this ok?
    w             : uint,
    h             : uint,
    tgt_fmt       : ColFmt,

    eoi_reached      : bool,
    has_frame_header : bool,

    qtables     : [[u8, ..64], ..4],
    ac_tables   : [HuffTab, ..2],
    dc_tables   : [HuffTab, ..2],

    cb          : u8,   // current byte
    bits_left   : uint, // num of unused bits in cb

    num_mcu_x   : uint,
    num_mcu_y   : uint,
    restart_interval : uint,

    comps       : [Component, ..3],
    index_for   : [uint, ..3],
    num_comps   : uint,
    hmax        : uint,
    vmax        : uint,
}

struct HuffTab {
    values  : [u8, ..256],
    sizes   : [u8, ..257],
    mincode : [i16, ..16],
    maxcode : [i16, ..16],
    valptr  : [i16, ..16],
}

struct Component {
    id       : u8,
    sfx      : uint,            // sampling factor, aka. h
    sfy      : uint,            // sampling factor, aka. v
    x        : uint,          // total number of samples without fill samples
    y        : uint,          // total number of samples without fill samples
    qtable   : uint,
    ac_table : uint,
    dc_table : uint,
    pred     : int,          // dc prediction
    data     : Vec<u8>,      // reconstructed samples
}

fn read_markers<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut has_next_scan_header = false;
    while !has_next_scan_header && !dc.eoi_reached {
        let mut marker: [u8, ..2] = [0, 0];
        try!(dc.stream.read_at_least(marker.len(), marker));

        if marker[0] != 0xff { return IFErr!("no marker"); }
        while marker[1] == 0xff {
            try!(dc.stream.read_at_least(1, marker[mut 1..2]));
        }

        //println!("marker: 0x{:x}", marker[1]);
        match marker[1] {
            DHT => try!(read_huffman_tables(dc)),
            DQT => try!(read_quantization_tables(dc)),
            SOF0 => {
                if dc.has_frame_header {
                    return IFErr!("extra frame header");
                }
                try!(read_frame_header(dc));
                dc.has_frame_header = true;
            }
            SOS => {
                if !dc.has_frame_header {
                    return IFErr!("no frame header");
                }
                try!(read_scan_header(dc));
                has_next_scan_header = true;
            }
            DRI => try!(read_restart_interval(dc)),
            EOI => dc.eoi_reached = true,
            APP0 ... APPF | COM => {
                //println!("skipping unknown marker...");
                let mut tmp: [u8, ..2] = [0, 0];
                try!(dc.stream.read_at_least(tmp.len(), tmp));
                let len = u16_from_be(tmp) - 2;
                try!(skip(dc.stream, len as uint));
            }
            _ => return IFErr!("invalid / unsupported marker"),
        }
    }
    Ok(())
}

//const SOI: u8 = 0xd8;     // start of image
const SOF0: u8 = 0xc0;    // start of frame / baseline DCT
//const SOF1: u8 = 0xc1;    // start of frame / extended seq.
const SOF2: u8 = 0xc2;    // start of frame / progressive DCT
//const SOF3: u8 = 0xc3;    // start of frame / lossless
//const SOF9: u8 = 0xc9;    // start of frame / extended seq., arithmetic
//const SOF11: u8 = 0xcb;    // start of frame / lossless, arithmetic
const DHT: u8 = 0xc4;     // define huffman tables
const DQT: u8 = 0xdb;     // define quantization tables
const DRI: u8 = 0xdd;     // define restart interval
const SOS: u8 = 0xda;     // start of scan
//const DNL: u8 = 0xdc;     // define number of lines
const RST0: u8 = 0xd0;    // restart entropy coded data
// ...
const RST7: u8 = 0xd7;    // restart entropy coded data
const APP0: u8 = 0xe0;    // application 0 segment
// ...
const APPF: u8 = 0xef;    // application f segment
//const DAC: u8 = 0xcc;     // define arithmetic conditioning table
const COM: u8 = 0xfe;     // comment
const EOI: u8 = 0xd9;     // end of image

fn read_huffman_tables<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8, ..17] = unsafe { zeroed() };
    try!(dc.stream.read_at_least(2, buf[mut 0..2]));
    let mut len = u16_from_be(buf[0..2]) as int -2;

    while 0 < len {
        try!(dc.stream.read_at_least(17, buf[mut 0..17]));  // info byte and BITS
        let table_class = buf[0] >> 4;            // 0 = dc table, 1 = ac table
        let table_slot = (buf[0] & 0xf) as uint;  // must be 0 or 1 for baseline
        if 1 < table_slot || 1 < table_class {
            return IFErr!("invalid / not supported");
        }

        // compute total number of huffman codes
        let mut mt = 0u;
        for i in range(1, 17) {
            mt += buf[i] as uint;
        }
        if 256 < mt {
            return IFErr!("invalid / not supported");
        }

        if table_class == 0 {
            try!(dc.stream.read_at_least(mt, dc.dc_tables[table_slot].values[mut 0..mt]));
            derive_table(&mut dc.dc_tables[table_slot], buf[1..17]);
        } else {
            try!(dc.stream.read_at_least(mt, dc.ac_tables[table_slot].values[mut 0..mt]));
            derive_table(&mut dc.ac_tables[table_slot], buf[1..17]);
        }

        len -= 17 + mt as int;
    }
    Ok(())
}

fn derive_table(table: &mut HuffTab, num_values: &[u8]) {
    assert!(num_values.len() == 16);

    let mut codes: [i16, ..256] = unsafe { zeroed() };

    let mut k = 0;
    for i in range(0, 16) {
        for j in range(0, num_values[i] as uint) {
            table.sizes[k] = (i + 1) as u8;
            k += 1;
        }
    }
    table.sizes[k] = 0;

    k = 0;
    let mut code = 0_i16;
    let mut si = table.sizes[k];
    loop {
        while si == table.sizes[k] {
            codes[k] = code;
            code += 1;
            k += 1;
        }

        if table.sizes[k] == 0 { break; }

        while si != table.sizes[k] {
            code <<= 1;
            si += 1;
        }
    }

    derive_mincode_maxcode_valptr(
        &mut table.mincode, &mut table.maxcode, &mut table.valptr,
        &codes, num_values
    );
}

fn derive_mincode_maxcode_valptr(mincode: &mut[i16, ..16], maxcode: &mut[i16, ..16],
                                 valptr:  &mut[i16, ..16], codes: &[i16, ..256],
                                 num_values: &[u8])
{
    for i in range(0, 16) {
        mincode[i] = -1;
        maxcode[i] = -1;
        valptr[i] = -1;
    }

    let mut j = 0u;
    for i in range(0, 16) {
        if num_values[i] != 0 {
            valptr[i] = j as i16;
            mincode[i] = codes[j];
            j += (num_values[i] - 1) as uint;
            maxcode[i] = codes[j];
            j += 1;
        }
    }
}

fn read_quantization_tables<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8, ..2] = unsafe { zeroed() };
    try!(dc.stream.read_at_least(2, buf[mut 0..2]));
    let mut len = u16_from_be(buf[0..2]) as uint -2;
    if len % 65 != 0 {
        return IFErr!("invalid / not supported");
    }

    while 0 < len {
        try!(dc.stream.read_at_least(1, buf[mut 0..1]));
        let precision = buf[0] >> 4;  // 0 = 8 bit, 1 = 16 bit
        let table_slot = (buf[0] & 0xf) as uint;
        if 3 < table_slot || precision != 0 {   // only 8 bit for baseline
            return IFErr!("invalid / not supported");
        }
        try!(dc.stream.read_at_least(64, &mut dc.qtables[table_slot]));
        len -= 65;
    }
    Ok(())
}

fn read_frame_header<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8, ..9] = unsafe { zeroed() };
    try!(dc.stream.read_at_least(8, buf[mut 0..8]));
    let len = u16_from_be(buf[0..2]) as uint;
    let precision = buf[2];
    dc.h = u16_from_be(buf[3..5]) as uint;
    dc.w = u16_from_be(buf[5..7]) as uint;
    dc.num_comps = buf[7] as uint;

    if precision != 8 || (dc.num_comps != 1 && dc.num_comps != 3) ||
       len != 8 + dc.num_comps*3 {
        return IFErr!("invalid / not supported");
    }

    dc.hmax = 0;
    dc.vmax = 0;
    let mut mcu_du = 0; // data units in one mcu
    try!(dc.stream.read_at_least(dc.num_comps*3, buf[mut 0..dc.num_comps*3]));

    for i in range(0, dc.num_comps) {
        let ci = (buf[i*3]-1) as uint;
        if dc.num_comps <= ci {
            return IFErr!("invalid / not supported");
        }
        dc.index_for[i] = ci;
        let sampling_factors = buf[i*3 + 1];
        let comp = &mut dc.comps[ci];
        *comp = Component {
            id      : buf[i*3],
            sfx     : (sampling_factors >> 4) as uint,
            sfy     : (sampling_factors & 0xf) as uint,
            x       : 0,
            y       : 0,
            qtable  : buf[i*3 + 2] as uint,
            ac_table : 0,
            dc_table : 0,
            pred    : 0,
            data    : Vec::from_elem(0, 0u8),
        };
        if comp.sfy < 1 || 4 < comp.sfy ||
           comp.sfx < 1 || 4 < comp.sfx ||
           3 < comp.qtable {
            return IFErr!("invalid / not supported");
        }

        if dc.hmax < comp.sfx { dc.hmax = comp.sfx; }
        if dc.vmax < comp.sfy { dc.vmax = comp.sfy; }
        mcu_du += comp.sfx * comp.sfy;
    }
    if 10 < mcu_du { return IFErr!("invalid / not supported"); }

    for i in range(0, dc.num_comps) {
        dc.comps[i].x = (dc.w as f64 * dc.comps[i].sfx as f64 / dc.hmax as f64).ceil() as uint;
        dc.comps[i].y = (dc.h as f64 * dc.comps[i].sfy as f64 / dc.vmax as f64).ceil() as uint;
    }

    let mcu_w = dc.hmax * 8;
    let mcu_h = dc.vmax * 8;
    dc.num_mcu_x = (dc.w + mcu_w-1) / mcu_w;
    dc.num_mcu_y = (dc.h + mcu_h-1) / mcu_h;

    Ok(())
}

fn read_scan_header<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8, ..3] = [0, 0, 0];
    try!(dc.stream.read_at_least(buf.len(), buf));
    let len = u16_from_be(buf[0..2]) as uint;
    let num_scan_comps = buf[2] as uint;

    if num_scan_comps != dc.num_comps || len != (6+num_scan_comps*2) {
        return IFErr!("invalid / not supported");
    }

    let mut compbuf = Vec::from_elem(len-3, 0u8);
    try!(dc.stream.read_at_least(compbuf.len(), compbuf[mut]));

    for i in range(0, num_scan_comps) {
        let comp_id = compbuf[i*2];
        let mut ci = 0;
        while ci < dc.num_comps && dc.comps[ci].id != comp_id { ci+=1 }
        if dc.num_comps <= ci {
            return IFErr!("invalid / not supported");
        }

        let tables = compbuf[i*2+1];
        dc.comps[ci].dc_table = (tables >> 4) as uint;
        dc.comps[ci].ac_table = (tables & 0xf) as uint;
        if 1 < dc.comps[ci].dc_table || 1 < dc.comps[i].ac_table {
            return IFErr!("invalid / not supported");
        }
    }

    // ignore last 3 bytes: spectral_start, spectral_end, approx
    Ok(())
}

fn read_restart_interval<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8, ..4] = [0, 0, 0, 0];
    try!(dc.stream.read_at_least(buf.len(), buf));
    let len = u16_from_be(buf[0..2]) as uint;
    if len != 4 { return IFErr!("invalid / not supported"); }
    dc.restart_interval = u16_from_be(buf[2..4]) as uint;
    Ok(())
}

fn decode_jpeg<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<Vec<u8>> {
    for i in range(0, dc.num_comps) {
        let comp = &mut dc.comps[i];
        comp.data = Vec::from_elem(dc.num_mcu_x*comp.sfx*8*dc.num_mcu_y*comp.sfy*8, 0u8);
    }

    // progressive images aren't supported so only one scan
    //println!("decode scan...");
    try!(decode_scan(dc));
    // throw away fill samples and convert to target format
    reconstruct(dc)
}

fn decode_scan<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let (mut intervals, mut mcus) =
        if 0 < dc.restart_interval {
            let total_mcus = dc.num_mcu_x * dc.num_mcu_y;
            let ivals = (total_mcus + dc.restart_interval-1) / dc.restart_interval;
            (ivals, dc.restart_interval)
        } else {
            (1, dc.num_mcu_x * dc.num_mcu_y)
        };

    for mcu_j in range(0, dc.num_mcu_y) {
        for mcu_i in range(0, dc.num_mcu_x) {

            // decode mcu
            for c in range(0, dc.num_comps) {
                let comp_idx = dc.index_for[c];
                let comp_sfx = dc.comps[comp_idx].sfx;
                let comp_sfy = dc.comps[comp_idx].sfy;
                let comp_qtab = dc.comps[comp_idx].qtable;

                for du_j in range(0, comp_sfy) {
                    for du_i in range(0, comp_sfx) {
                        // decode entropy, dequantize & dezigzag
                        //let data = try!(decode_block(dc, comp, &dc.qtables[comp.qtable]));
                        let data = try!(decode_block(dc, comp_idx, comp_qtab));

                        // idct & level-shift
                        let outx = (mcu_i * comp_sfx + du_i) * 8;
                        let outy = (mcu_j * comp_sfy + du_j) * 8;
                        let dst_stride = dc.num_mcu_x * comp_sfx * 8;
                        let base =
                            &mut dc.comps[comp_idx].data[mut][0] as *mut u8;
                        let offset = (outy * dst_stride + outx) as int;
                        unsafe {
                            let dst = base.offset(offset);
                            stbi_idct_block(dst, dst_stride, data);
                        }
                    }
                }
            }

            mcus -= 1;

            if mcus == 0 {
                intervals -= 1;
                if intervals == 0 {
                    return Ok(());
                }

                try!(read_restart(dc.stream));    // RSTx marker

                if intervals == 1 {
                    mcus = (dc.num_mcu_y-mcu_j-1) * dc.num_mcu_x + dc.num_mcu_x-mcu_i-1;
                } else {
                    mcus = dc.restart_interval;
                }

                // reset decoder
                dc.cb = 0;
                dc.bits_left = 0;
                for k in range(0, dc.num_comps) {
                    dc.comps[k].pred = 0;
                }
            }
        }
    }
    Ok(())
}

fn read_restart<R: Reader>(stream: &mut R) -> IoResult<()> {
    let mut buf: [u8, ..2] = [0, 0];
    try!(stream.read_at_least(buf.len(), buf));
    if buf[0] != 0xff || buf[1] < RST0 || RST7 < buf[1] {
        return IFErr!("reset marker missing");
    }
    Ok(())
}

static DEZIGZAG: [u8, ..64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// decode entropy, dequantize & dezigzag (section F.2)
//fn decode_block<R: Reader>(dc: &mut JpegDecoder<R>, comp: &mut Component,
//                                                     qtable: &[u8, ..64])
//                                                 -> IoResult<[i16, ..64]>
fn decode_block<R: Reader>(dc: &mut JpegDecoder<R>, comp_idx: uint, qtable_idx: uint)
                                                             -> IoResult<[i16, ..64]>
{
    //let comp = &mut dc.comps[comp_idx];
    //let qtable = &dc.qtables[qtable_idx];

    let mut res: [i16, ..64] = unsafe { zeroed() };
    //let t = try!(decode_huff(dc, dc.dc_tables[comp.dc_table]));
    let dc_table_idx = dc.comps[comp_idx].dc_table;
    let ac_table_idx = dc.comps[comp_idx].ac_table;
    let t = try!(decode_huff(dc, dc_table_idx, true));
    let diff: int = if 0 < t { try!(receive_and_extend(dc, t)) } else { 0 };

    dc.comps[comp_idx].pred += diff;
    res[0] = (dc.comps[comp_idx].pred * dc.qtables[qtable_idx][0] as int) as i16;

    let mut k = 1u;
    while k < 64 {
        //let rs = try!(decode_huff(dc, &dc.ac_tables[comp.ac_table]));
        let rs = try!(decode_huff(dc, ac_table_idx, false));
        let rrrr = rs >> 4;
        let ssss = rs & 0xf;

        if ssss == 0 {
            if rrrr != 0xf {
                break;  // end of block
            }
            k += 16;    // run length is 16
            continue;
        }
        k += rrrr as uint;

        if 63 < k {
            return IFErr!("corrupt block");
        }
        res[DEZIGZAG[k] as uint] =
            (try!(receive_and_extend(dc, ssss)) * dc.qtables[qtable_idx][k] as int) as i16;
        k += 1;
    }

    Ok(res)
}

//fn decode_huff<R: Reader>(dc: &mut JpegDecoder<R>, tab: &HuffTab) -> IoResult<u8> {
fn decode_huff<R: Reader>(dc: &mut JpegDecoder<R>, tab_idx: uint, dctab: bool) -> IoResult<u8> {
    let tab = & if dctab { dc.dc_tables[tab_idx] } else { dc.ac_tables[tab_idx] };

    let mut code = try!(nextbit(dc)) as i16;
    let mut i = 0;
    while tab.maxcode[i] < code {
        code = (code << 1) + try!(nextbit(dc)) as i16;
        i += 1;
        if tab.maxcode.len() <= i {
            return IFErr!("corrupt huffman coding");
        }
    }
    let j = (tab.valptr[i] + code - tab.mincode[i]) as uint;
    if tab.values.len() <= j {
        return IFErr!("corrupt huffman coding")
    }
    Ok(tab.values[j])
}

fn receive_and_extend<R: Reader>(dc: &mut JpegDecoder<R>, s: u8) -> IoResult<int> {
    // receive
    let mut symbol = 0i;
    for _ in range(0, s) {
        symbol = (symbol << 1) + try!(nextbit(dc)) as int;
    }
    // extend
    let vt = 1 << (s as uint - 1);
    if symbol < vt {
        Ok(symbol + (-1 << s as uint) + 1)
    } else {
        Ok(symbol)
    }
}

fn nextbit<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<u8> {
    if dc.bits_left == 0 {
        dc.cb = try!(dc.stream.read_u8());
        dc.bits_left = 8;

        if dc.cb == 0xff {
            let b2 = try!(dc.stream.read_u8());
            if b2 != 0x0 {
                return IFErr!("unexpected marker")
            }
        }
    }

    let r = dc.cb >> 7;
    dc.cb <<= 1;
    dc.bits_left -= 1;
    Ok(r)
}

fn reconstruct<R: Reader>(dc: &JpegDecoder<R>) -> IoResult<Vec<u8>> {
    let tgt_chans = dc.tgt_fmt.channels();
    let mut result = Vec::from_elem(dc.w * dc.h * tgt_chans, 0u8);

    match (dc.num_comps, dc.tgt_fmt) {
        (3, FmtRGB) | (3, FmtRGBA) => {
            for ref comp in dc.comps.iter() {
                if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                    upsample_rgb(dc, result[mut]);
                    return Ok(result);
                }
            }

            let mut si = 0u;
            let mut di = 0u;
            for _j in range(0, dc.h) {
                for i in range(0, dc.w) {
                    let pixel = ycbcr_to_rgb(
                        dc.comps[0].data[si+i],
                        dc.comps[1].data[si+i],
                        dc.comps[2].data[si+i],
                    );
                    copy_memory(result[mut di..di+3], pixel[]);
                    if dc.tgt_fmt == FmtRGBA { *result.get_mut(di+3) = 255; }
                    di += tgt_chans;
                }
                si += dc.num_mcu_x * dc.comps[0].sfx * 8;
            }
            return Ok(result);
        },
        (_, FmtY) | (_, FmtYA) => {
            let comp = &dc.comps[0];
            if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                upsample_gray(dc, result[mut]);
                return Ok(result);
            }

            // no resampling
            let mut si = 0u;
            let mut di = 0u;
            if dc.tgt_fmt == FmtYA {
                for _j in range(0, dc.h) {
                    for i in range(0, dc.w) {
                        *result.get_mut(di  ) = comp.data[si+i];
                        *result.get_mut(di+1) = 255;
                        di += 2;
                    }
                    si += dc.num_mcu_x * comp.sfx * 8;
                }
            } else {    // FmtY
                for _j in range(0, dc.h) {
                    copy_memory(result[mut di..di+dc.w], comp.data[si..si+dc.w]);
                    si += dc.num_mcu_x * comp.sfx * 8;
                    di += dc.w;
                }
            }
            return Ok(result);
        },
        (1, FmtRGB) | (1, FmtRGBA) => {
            let comp = &dc.comps[0];
            let mut si = 0u;
            let mut di = 0u;
            for _j in range(0, dc.h) {
                for i in range(0, dc.w) {
                    *result.get_mut(di  ) = comp.data[si+i];
                    *result.get_mut(di+1) = comp.data[si+i];
                    *result.get_mut(di+2) = comp.data[si+i];
                    if dc.tgt_fmt == FmtRGBA { *result.get_mut(di+3) = 255; }
                    di += tgt_chans;
                }
                si += dc.num_mcu_x * comp.sfx * 8;
            }
            return Ok(result);
        },
        _ => return IFErr!("internal error"),
    }
}

fn upsample_gray<R: Reader>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let mut di = 0u;
    let tgt_chans = dc.tgt_fmt.channels();

    for j in range(0, dc.h) {
        let si0 = (j as f64 * si0yratio).floor() as uint * stride0;
        for i in range(0, dc.w) {
            result[di] = dc.comps[0].data[si0 + (i as f64 * si0xratio).floor() as uint];
            if dc.tgt_fmt == FmtYA { result[di+1] = 255; }
            di += tgt_chans;
        }
    }
}

fn upsample_rgb<R: Reader>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let stride1 = dc.num_mcu_x * dc.comps[1].sfx * 8;
    let stride2 = dc.num_mcu_x * dc.comps[2].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si1yratio = dc.comps[1].y as f64 / dc.h as f64;
    let si2yratio = dc.comps[2].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let si1xratio = dc.comps[1].x as f64 / dc.w as f64;
    let si2xratio = dc.comps[2].x as f64 / dc.w as f64;

    let mut di = 0u;
    let tgt_chans = dc.tgt_fmt.channels();

    for j in range(0, dc.h) {
        let si0 = (j as f64 * si0yratio).floor() as uint * stride0;
        let si1 = (j as f64 * si1yratio).floor() as uint * stride1;
        let si2 = (j as f64 * si2yratio).floor() as uint * stride2;

        for i in range(0, dc.w) {
            let pixel = ycbcr_to_rgb(
                dc.comps[0].data[si0 + (i as f64 * si0xratio).floor() as uint],
                dc.comps[1].data[si1 + (i as f64 * si1xratio).floor() as uint],
                dc.comps[2].data[si2 + (i as f64 * si2xratio).floor() as uint],
            );
            copy_memory(result[mut di..di+3], pixel[]);
            if dc.tgt_fmt == FmtRGBA { result[di+3] = 255; }
            di += tgt_chans;
        }
    }
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8, ..3] {
    let cb = cb as f32;
    let cr = cr as f32;
    [clamp_to_u8(y as f32 + 1.402*(cr-128_f32)),
     clamp_to_u8(y as f32 - 0.34414*(cb-128_f32) - 0.71414*(cr-128_f32)),
     clamp_to_u8(y as f32 + 1.772*(cb-128_f32))]
}

fn clamp_to_u8(x: f32) -> u8 {
    if x < 0.0 { return 0; }
    if 255.0 < x { return 255; }
    x as u8
}

// ------------------------------------------------------------
// The IDCT stuff here (to the next dashed line) is copied and adapted from
// stb_image which is released under public domain.  Many thanks to stb_image
// author, Sean Barrett.
// Link: https://github.com/nothings/stb/blob/master/stb_image.h

// idct and level-shift
unsafe fn stbi_idct_block(mut dst: *mut u8, dst_stride: uint, data: &[i16]) {
    assert!(data.len() == 64);  // sigh
    let d = data;
    let mut v: [int, ..64] = zeroed();

    // columns
    for i in range(0, 8) {
        if d[i+ 8]==0 && d[i+16]==0 && d[i+24]==0 && d[i+32]==0 &&
           d[i+40]==0 && d[i+48]==0 && d[i+56]==0 {
            let dcterm = d[i] as int << 2;
            v[i   ] = dcterm;
            v[i+ 8] = dcterm;
            v[i+16] = dcterm;
            v[i+24] = dcterm;
            v[i+32] = dcterm;
            v[i+40] = dcterm;
            v[i+48] = dcterm;
            v[i+56] = dcterm;
        } else {
            let mut t0: int = 0; let mut t1: int = 0;
            let mut t2: int = 0; let mut t3: int = 0;
            let mut x0: int = 0; let mut x1: int = 0;
            let mut x2: int = 0; let mut x3: int = 0;
            stbi_idct_1d(
                &mut t0, &mut t1, &mut t2, &mut t3,
                &mut x0, &mut x1, &mut x2, &mut x3,
                d[i+ 0] as int, d[i+ 8] as int, d[i+16] as int, d[i+24] as int,
                d[i+32] as int, d[i+40] as int, d[i+48] as int, d[i+56] as int
            );

            // constants scaled things up by 1<<12; let's bring them back
            // down, but keep 2 extra bits of precision
            x0 += 512; x1 += 512; x2 += 512; x3 += 512;
            v[i+ 0] = (x0+t3) >> 10;
            v[i+56] = (x0-t3) >> 10;
            v[i+ 8] = (x1+t2) >> 10;
            v[i+48] = (x1-t2) >> 10;
            v[i+16] = (x2+t1) >> 10;
            v[i+40] = (x2-t1) >> 10;
            v[i+24] = (x3+t0) >> 10;
            v[i+32] = (x3-t0) >> 10;
        }
    }

    for i in range_step(0, 64, 8) {
        let mut t0: int = 0; let mut t1: int = 0;
        let mut t2: int = 0; let mut t3: int = 0;
        let mut x0: int = 0; let mut x1: int = 0;
        let mut x2: int = 0; let mut x3: int = 0;
        stbi_idct_1d(
            &mut t0, &mut t1, &mut t2, &mut t3,
            &mut x0, &mut x1, &mut x2, &mut x3,
            v[i+0],v[i+1],v[i+2],v[i+3],v[i+4],v[i+5],v[i+6],v[i+7]
        );
        // constants scaled things up by 1<<12, plus we had 1<<2 from first
        // loop, plus horizontal and vertical each scale by sqrt(8) so together
        // we've got an extra 1<<3, so 1<<17 total we need to remove.
        // so we want to round that, which means adding 0.5 * 1<<17,
        // aka 65536. Also, we'll end up with -128 to 127 that we want
        // to encode as 0-255 by adding 128, so we'll add that before the shift
        x0 += 65536 + (128<<17);
        x1 += 65536 + (128<<17);
        x2 += 65536 + (128<<17);
        x3 += 65536 + (128<<17);

        *dst.offset(0) = stbi_clamp((x0+t3) >> 17);
        *dst.offset(7) = stbi_clamp((x0-t3) >> 17);
        *dst.offset(1) = stbi_clamp((x1+t2) >> 17);
        *dst.offset(6) = stbi_clamp((x1-t2) >> 17);
        *dst.offset(2) = stbi_clamp((x2+t1) >> 17);
        *dst.offset(5) = stbi_clamp((x2-t1) >> 17);
        *dst.offset(3) = stbi_clamp((x3+t0) >> 17);
        *dst.offset(4) = stbi_clamp((x3-t0) >> 17);

        dst = dst.offset(dst_stride as int);
    }
}

fn stbi_clamp(x: int) -> u8 {
   if x as uint > 255 {
      if x < 0 { return 0; }
      if x > 255 { return 255; }
   }
   return x as u8;
}

fn stbi_idct_1d(t0: &mut int, t1: &mut int, t2: &mut int, t3: &mut int,
                 x0: &mut int, x1: &mut int, x2: &mut int, x3: &mut int,
        s0: int, s1: int, s2: int, s3: int, s4: int, s5: int, s6: int, s7: int)
{
   let mut p2 = s2;
   let mut p3 = s6;
   let mut p1 = (p2+p3) * f2f(0.5411961_f32);
   *t2 = p1 + p3 * f2f(-1.847759065_f32);
   *t3 = p1 + p2 * f2f( 0.765366865_f32);
   p2 = s0;
   p3 = s4;
   *t0 = fsh(p2+p3);
   *t1 = fsh(p2-p3);
   *x0 = *t0+*t3;
   *x3 = *t0-*t3;
   *x1 = *t1+*t2;
   *x2 = *t1-*t2;
   *t0 = s7;
   *t1 = s5;
   *t2 = s3;
   *t3 = s1;
   p3 = *t0+*t2;
   let mut p4 = *t1+*t3;
   p1 = *t0+*t3;
   p2 = *t1+*t2;
   let p5 = (p3+p4)*f2f( 1.175875602_f32);
   *t0 = *t0*f2f( 0.298631336_f32);
   *t1 = *t1*f2f( 2.053119869_f32);
   *t2 = *t2*f2f( 3.072711026_f32);
   *t3 = *t3*f2f( 1.501321110_f32);
   p1 = p5 + p1*f2f(-0.899976223_f32);
   p2 = p5 + p2*f2f(-2.562915447_f32);
   p3 = p3*f2f(-1.961570560_f32);
   p4 = p4*f2f(-0.390180644_f32);
   *t3 += p1+p4;
   *t2 += p2+p3;
   *t1 += p2+p4;
   *t0 += p1+p3;
}

#[inline(always)] fn f2f(x: f32) -> int { (x * 4096_f32 + 0.5) as int }
#[inline(always)] fn fsh(x: int) -> int { x << 12 }

// ------------------------------------------------------------

type LineConverter = fn(&[u8], &mut[u8]);

fn get_converter(src_fmt: ColFmt, tgt_fmt: ColFmt) -> IoResult<LineConverter> {
    match (src_fmt, tgt_fmt) {
        (s, t) if (s == t) => Ok(copy_line),
        (FmtY, FmtYA)      => Ok(y_to_ya),
        (FmtY, FmtRGB)     => Ok(y_to_rgb),
        (FmtY, FmtRGBA)    => Ok(y_to_rgba),
        (FmtY, FmtBGR)     => Ok(y_to_bgr),
        (FmtY, FmtBGRA)    => Ok(y_to_bgra),
        (FmtYA, FmtY)      => Ok(ya_to_y),
        (FmtYA, FmtRGB)    => Ok(ya_to_rgb),
        (FmtYA, FmtRGBA)   => Ok(ya_to_rgba),
        (FmtYA, FmtBGR)    => Ok(ya_to_bgr),
        (FmtYA, FmtBGRA)   => Ok(ya_to_bgra),
        (FmtRGB, FmtY)     => Ok(rgb_to_y),
        (FmtRGB, FmtYA)    => Ok(rgb_to_ya),
        (FmtRGB, FmtRGBA)  => Ok(rgb_to_rgba),
        (FmtRGB, FmtBGR)   => Ok(rgb_to_bgr),
        (FmtRGB, FmtBGRA)  => Ok(rgb_to_bgra),
        (FmtRGBA, FmtY)    => Ok(rgba_to_y),
        (FmtRGBA, FmtYA)   => Ok(rgba_to_ya),
        (FmtRGBA, FmtRGB)  => Ok(rgba_to_rgb),
        (FmtRGBA, FmtBGR)  => Ok(rgba_to_bgr),
        (FmtRGBA, FmtBGRA) => Ok(rgba_to_bgra),
        (FmtBGR, FmtY)     => Ok(bgr_to_y),
        (FmtBGR, FmtYA)    => Ok(bgr_to_ya),
        (FmtBGR, FmtRGB)   => Ok(bgr_to_rgb),
        (FmtBGR, FmtRGBA)  => Ok(bgr_to_rgba),
        (FmtBGRA, FmtY)    => Ok(bgra_to_y),
        (FmtBGRA, FmtYA)   => Ok(bgra_to_ya),
        (FmtBGRA, FmtRGB)  => Ok(bgra_to_rgb),
        (FmtBGRA, FmtRGBA) => Ok(bgra_to_rgba),
        _ => IFErr!("conversion not supported"),
    }
}

fn copy_line(src_line: &[u8], tgt_line: &mut[u8]) {
    copy_memory(tgt_line, src_line);
}

fn luminance(r: u8, g: u8, b: u8) -> u8 {
    (0.21 * r as f32 + 0.64 * g as f32 + 0.15 * b as f32) as u8
}

fn y_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range(0u, src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = 255;
        t += 2;
    }
}

static y_to_bgr: LineConverter = y_to_rgb;
fn y_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range(0u, src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        t += 3;
    }
}

static y_to_bgra: LineConverter = y_to_rgba;
fn y_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range(0u, src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn ya_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t] = src_line[s];
        t += 1;
    }
}

static ya_to_bgr: LineConverter = ya_to_rgb;
fn ya_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        t += 3;
    }
}

static ya_to_bgra: LineConverter = ya_to_rgba;
fn ya_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn rgb_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        t += 1;
    }
}

fn rgb_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = 255;
        t += 2;
    }
}

fn rgb_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn rgba_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        t += 1;
    }
}

fn rgba_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = src_line[s+3];
        t += 2;
    }
}

fn rgba_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        t += 3;
    }
}

fn bgr_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        t += 1;
    }
}

fn bgr_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = 255;
        t += 2;
    }
}

static rgb_to_bgr: LineConverter = bgr_to_rgb;
fn bgr_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        t += 3;
    }
}

static rgb_to_bgra: LineConverter = bgr_to_rgba;
fn bgr_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn bgra_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        t += 1;
    }
}

fn bgra_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = src_line[s+3];
        t += 2;
    }
}

static rgba_to_bgr: LineConverter = bgra_to_rgb;
fn bgra_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        t += 3;
    }
}

static rgba_to_bgra: LineConverter = bgra_to_rgba;
fn bgra_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0u;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        tgt_line[t+3] = src_line[s+3];
        t += 4;
    }
}

// ------------------------------------------------------------

fn crc32be(data: &[u8]) -> [u8, ..4] {
    Crc32::new().put(data).finish_be()
}

struct Crc32 { r: u32 }
impl Crc32 {
    fn new() -> Crc32 { Crc32 { r: 0xffff_ffff } }

    fn put<'a>(&'a mut self, bytes: &[u8]) -> &'a mut Crc32 {
        for byte in bytes.iter() {
            let idx = byte ^ (self.r as u8);
            self.r = (self.r >> 8) ^ CRC32_TABLE[idx as uint];
        }
        self
    }

    fn finish_be(&mut self) -> [u8, ..4] {
        let result = u32_to_be(self.r ^ 0xffff_ffff);
        self.r = 0xffff_ffff;
        result
    }
}

static CRC32_TABLE: [u32, ..256] = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
    0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
    0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
    0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
    0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
    0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
    0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
    0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
    0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
    0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
    0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
    0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
    0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
    0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
    0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
    0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
    0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
    0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
    0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
    0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
    0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
    0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
];

// ------------------------------------------------------------

fn u16_from_be(buf: &[u8]) -> u16 {
    buf[0] as u16 << 8 | buf[1] as u16
}

fn u16_from_le(buf: &[u8]) -> u16 {
    buf[1] as u16 << 8 | buf[0] as u16
}

fn u16_to_le(x: u16) -> [u8, ..2] {
    let buf = [x as u8, (x >> 8) as u8];
    buf
}

fn u32_from_be(buf: &[u8]) -> u32 {
    buf[0] as u32 << 24 | buf[1] as u32 << 16 | buf[2] as u32 << 8 | buf[3] as u32
}

fn u32_to_be(x: u32) -> [u8, ..4] {
    let buf = [(x >> 24) as u8, (x >> 16) as u8,
               (x >>  8) as u8, (x)       as u8];
    buf
}

fn equal(a: &[u8], b: &[u8]) -> bool {
    for i in range(0, a.len()) {
        if a[i] != b[i] { return false; }
    }
    true
}

fn skip<R: Reader>(stream: &mut R, mut bytes: uint) -> IoResult<()> {
    let mut buf: [u8, ..1024] = unsafe { zeroed() };
    while 0 < bytes {
        let n = min(bytes, buf.len());
        try!(stream.read_at_least(n, buf[mut 0..n]));
        bytes -= n;
    }
    Ok(())
}

fn extract_extension(filename: &str) -> Option<&str> {
    match filename.rfind('.') {
        Some(i) => Some(filename[i..]),
        None => None,
    }
}
