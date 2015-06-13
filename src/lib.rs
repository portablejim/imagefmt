// Copyright (c) 2014-2015 Tero Hänninen
//
//! # Example
//!
//! ```no_run
//! extern crate imagefmt;
//! use imagefmt::{ColFmt, ColType};
//!
//! fn main() {
//!     // load and convert to bgra
//!     let _pic = imagefmt::read("stars.jpg", ColFmt::BGRA).unwrap();
//!
//!     // convert to one of y, ya, rgb, rgba
//!     let pic = imagefmt::read("marbles.tga", ColFmt::Auto).unwrap();
//!
//!     // write image out as grayscale
//!     pic.write("out.png", ColType::Gray).unwrap();
//!
//!     // there's also a free function that doesn't require an Image
//!     imagefmt::write("out.tga", pic.w, pic.h, pic.fmt, &pic.buf,
//!                                                  ColType::Gray)
//!                                                      .unwrap();
//!
//!     // get width, height and color type
//!     let _info = imagefmt::read_info("hiisi.png").unwrap();
//! }
//! ```

use std::ffi::OsStr;
use std::fs::{File};
use std::io::{self, Read, BufReader, BufWriter, ErrorKind};
use std::iter::{repeat};
use std::path::Path;
use std::fmt::{self, Debug};
use std::ptr;

mod png;
mod tga;
mod bmp;
mod jpeg;

pub use png::{read_png, read_png_info, read_png_chunks,
              write_png, write_png_chunks, PngCustomChunk};
pub use tga::{read_tga, read_tga_info, write_tga};
pub use bmp::{read_bmp, read_bmp_info};
pub use jpeg::{read_jpeg, read_jpeg_info};

/// Functions for reading headers, stuff like that.
// Private for now, at least.
mod ext {
    pub use super::png::{read_png_header, PngHeader};
    pub use super::tga::{read_tga_header, TgaHeader};
    pub use super::bmp::{read_bmp_header, BmpHeader, DibV1, DibV2, DibV4, DibV5};
}

/// Image struct returned from the read functions.
#[derive(Clone)]
pub struct Image {
    pub w   : usize,
    pub h   : usize,
    pub fmt : ColFmt,
    pub buf : Vec<u8>,
}

/// Holds basic info about an image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Info {
    pub w : usize,
    pub h : usize,
    pub ct : ColType,
}

/// Color format.
///
/// `Auto` means automatic/infer
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColFmt {
    Auto,
    Y,
    YA,
    RGB,
    RGBA,
    BGR,
    BGRA,
}

/// Color type – these are categories of color formats.
///
/// `Auto` means automatic/infer/unknown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColType {
    Auto,
    Gray,
    GrayAlpha,
    Color,
    ColorAlpha,
}

/// Returns width, height and color type of the image.
pub fn read_info<P: AsRef<Path>>(filepath: P) -> io::Result<Info> {
    let filepath: &Path = filepath.as_ref();
    type F = fn(&mut BufReader<File>) -> io::Result<Info>;
    let readfunc: F =
        match filepath.extension().and_then(OsStr::to_str) {
            Some("png")                => read_png_info,
            Some("tga")                => read_tga_info,
            Some("bmp")                => read_bmp_info,
            Some("jpg") | Some("jpeg") => read_jpeg_info,
            _ => return error("extension not recognized"),
        };
    let file = try!(File::open(filepath));
    let reader = &mut BufReader::new(file);
    readfunc(reader)
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to one of `Y`, `YA`, `RGB`,
/// `RGBA`.  Paletted images are auto-depaletted.
pub fn read<P: AsRef<Path>>(filepath: P, req_fmt: ColFmt) -> io::Result<Image> {
    let filepath: &Path = filepath.as_ref();
    type F = fn(&mut BufReader<File>, ColFmt) -> io::Result<Image>;
    let readfunc: F =
        match filepath.extension().and_then(OsStr::to_str) {
            Some("png")                => read_png,
            Some("tga")                => read_tga,
            Some("bmp")                => read_bmp,
            Some("jpg") | Some("jpeg") => read_jpeg,
            _ => return error("extension not recognized"),
        };
    let file = try!(File::open(filepath));
    let reader = &mut BufReader::new(file);
    readfunc(reader, req_fmt)
}

/// Writes an image and converts it to requested color type.
pub fn write<P>(filepath: P, w: usize, h: usize, src_fmt: ColFmt, data: &[u8],
                                                            tgt_type: ColType)
                                                             -> io::Result<()>
        where P: AsRef<Path>
{
    let filepath: &Path = filepath.as_ref();
    type F = fn(&mut BufWriter<File>, usize, usize, ColFmt, &[u8], ColType)
                                                         -> io::Result<()>;
    let writefunc: F =
        match filepath.extension().and_then(OsStr::to_str) {
            Some("png") => write_png,
            Some("tga") => write_tga,
            _ => return error("extension not supported for writing"),
        };
    let file = try!(File::create(filepath));
    let writer = &mut BufWriter::new(file);
    writefunc(writer, w, h, src_fmt, data, tgt_type)
}

/// Converts the image into a new allocation.
pub fn convert(w: usize, h: usize, src_fmt: ColFmt, data: &[u8], tgt_fmt: ColFmt)
                                                           -> Result<Image, &str>
{
    let src_bytespp = data.len() / w / h;

    if w < 1 || h < 1
    || src_bytespp * w * h != data.len()
    || src_bytespp != src_fmt.bytes_pp() {
        return Err("invalid dimensions or data length");
    }

    if src_fmt == ColFmt::Auto {
        return Err("can't convert from unknown source format")
    }

    if tgt_fmt == src_fmt || tgt_fmt == ColFmt::Auto {
        let mut result: Vec<u8> = repeat(0).take(data.len()).collect();
        copy_memory(data, &mut result[..]);
        return Ok(Image {
            w   : w,
            h   : h,
            fmt : src_fmt,
            buf : result,
        })
    }

    let convert = try!(converter(src_fmt, tgt_fmt).map_err(|_| "no such converter"));

    let src_linesize = w * src_fmt.bytes_pp();
    let tgt_linesize = w * tgt_fmt.bytes_pp();
    let mut result: Vec<u8> = repeat(0).take(h * tgt_linesize).collect();

    let mut si = 0;
    let mut ti = 0;
    for _j in (0 .. h) {
        convert(&data[si .. si+src_linesize], &mut result[ti .. ti+tgt_linesize]);
        si += src_linesize;
        ti += tgt_linesize;
    }

    Ok(Image {
        w   : w,
        h   : h,
        fmt : tgt_fmt,
        buf : result,
    })
}

impl Image {
    /// Writes an image and converts it to requested color type.
    #[inline]
    pub fn write<P>(&self, filepath: P, tgt_type: ColType) -> io::Result<()>
            where P: AsRef<Path>
    {
        write(filepath, self.w, self.h, self.fmt, &self.buf, tgt_type)
    }

    /// Converts the image into a new allocation.
    #[inline]
    pub fn convert(&self, tgt_fmt: ColFmt) -> Result<Image, &str> {
        convert(self.w, self.h, self.fmt, &self.buf, tgt_fmt)
    }
}

impl ColFmt {
    /// Returns the color type of the color format.
    pub fn color_type(self) -> ColType {
        match self {
            ColFmt::Y => ColType::Gray,
            ColFmt::YA => ColType::GrayAlpha,
            ColFmt::RGB => ColType::Color,
            ColFmt::RGBA => ColType::ColorAlpha,
            ColFmt::BGR => ColType::Color,
            ColFmt::BGRA => ColType::ColorAlpha,
            ColFmt::Auto => ColType::Auto,
        }
    }

    fn bytes_pp(&self) -> usize {
        use self::ColFmt::*;
        match *self {
            Auto        => 0,
            Y           => 1,
            YA          => 2,
            RGB  | BGR  => 3,
            RGBA | BGRA => 4,
        }
    }
}

// ------------------------------------------------------------

type LineConverter = fn(&[u8], &mut[u8]);

fn converter(src_fmt: ColFmt, tgt_fmt: ColFmt) -> io::Result<LineConverter> {
    use self::ColFmt::*;
    match (src_fmt, tgt_fmt) {
        (ref s, ref t) if (*s == *t) => Ok(copy_memory),
        (Y, YA)      => Ok(y_to_ya),
        (Y, RGB)     => Ok(y_to_rgb),
        (Y, RGBA)    => Ok(y_to_rgba),
        (Y, BGR)     => Ok(Y_TO_BGR),
        (Y, BGRA)    => Ok(Y_TO_BGRA),
        (YA, Y)      => Ok(ya_to_y),
        (YA, RGB)    => Ok(ya_to_rgb),
        (YA, RGBA)   => Ok(ya_to_rgba),
        (YA, BGR)    => Ok(YA_TO_BGR),
        (YA, BGRA)   => Ok(YA_TO_BGRA),
        (RGB, Y)     => Ok(rgb_to_y),
        (RGB, YA)    => Ok(rgb_to_ya),
        (RGB, RGBA)  => Ok(rgb_to_rgba),
        (RGB, BGR)   => Ok(RGB_TO_BGR),
        (RGB, BGRA)  => Ok(RGB_TO_BGRA),
        (RGBA, Y)    => Ok(rgba_to_y),
        (RGBA, YA)   => Ok(rgba_to_ya),
        (RGBA, RGB)  => Ok(rgba_to_rgb),
        (RGBA, BGR)  => Ok(RGBA_TO_BGR),
        (RGBA, BGRA) => Ok(RGBA_TO_BGRA),
        (BGR, Y)     => Ok(bgr_to_y),
        (BGR, YA)    => Ok(bgr_to_ya),
        (BGR, RGB)   => Ok(bgr_to_rgb),
        (BGR, RGBA)  => Ok(bgr_to_rgba),
        (BGR, BGRA)  => Ok(BGR_TO_BGRA),
        (BGRA, Y)    => Ok(bgra_to_y),
        (BGRA, YA)   => Ok(bgra_to_ya),
        (BGRA, RGB)  => Ok(bgra_to_rgb),
        (BGRA, RGBA) => Ok(bgra_to_rgba),
        (BGRA, BGR)  => Ok(BGRA_TO_BGR),
        _ => error("no such converter"),
    }
}

fn luminance(r: u8, g: u8, b: u8) -> u8 {
    (0.21 * r as f32 + 0.64 * g as f32 + 0.15 * b as f32) as u8
}

fn y_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0;
    for &sb in src_line {
        tgt_line[t  ] = sb;
        tgt_line[t+1] = 255;
        t += 2;
    }
}

const Y_TO_BGR: LineConverter = y_to_rgb;
fn y_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0;
    for &sb in src_line {
        tgt_line[t  ] = sb;
        tgt_line[t+1] = sb;
        tgt_line[t+2] = sb;
        t += 3;
    }
}

const Y_TO_BGRA: LineConverter = y_to_rgba;
fn y_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0;
    for &sb in src_line {
        tgt_line[t  ] = sb;
        tgt_line[t+1] = sb;
        tgt_line[t+2] = sb;
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn ya_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    for tb in tgt_line {
        *tb = src_line[s];
        s += 2;
    }
}

const YA_TO_BGR: LineConverter = ya_to_rgb;
fn ya_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        s += 2;
        t += 3;
    }
}

const YA_TO_BGRA: LineConverter = ya_to_rgba;
fn ya_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        tgt_line[t+3] = src_line[s+1];
        s += 2;
        t += 4;
    }
}

fn rgb_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    for tb in tgt_line {
        *tb = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        s += 3;
    }
}

fn rgb_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = 255;
        s += 3;
        t += 2;
    }
}

const BGR_TO_BGRA: LineConverter = rgb_to_rgba;
fn rgb_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        tgt_line[t+3] = 255;
        s += 3;
        t += 4;
    }
}

fn rgba_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    for tb in tgt_line {
        *tb = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        s += 4;
    }
}

fn rgba_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = src_line[s+3];
        s += 4;
        t += 2;
    }
}

const BGRA_TO_BGR: LineConverter = rgba_to_rgb;
fn rgba_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        s += 4;
        t += 3;
    }
}

fn bgr_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    for tb in tgt_line {
        *tb = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        s += 3;
    }
}

fn bgr_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = 255;
        s += 3;
        t += 2;
    }
}

const RGB_TO_BGR: LineConverter = bgr_to_rgb;
fn bgr_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut st = 0;
    while st < src_line.len() {
        tgt_line[st  ] = src_line[st+2];
        tgt_line[st+1] = src_line[st+1];
        tgt_line[st+2] = src_line[st  ];
        st += 3;
    }
}

const RGB_TO_BGRA: LineConverter = bgr_to_rgba;
fn bgr_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        tgt_line[t+3] = 255;
        s += 3;
        t += 4;
    }
}

fn bgra_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    for tb in tgt_line {
        *tb = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        s += 4;
    }
}

fn bgra_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = src_line[s+3];
        s += 4;
        t += 2;
    }
}

const RGBA_TO_BGR: LineConverter = bgra_to_rgb;
fn bgra_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut s = 0;
    let mut t = 0;
    while s < src_line.len() {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        s += 4;
        t += 3;
    }
}

const RGBA_TO_BGRA: LineConverter = bgra_to_rgba;
fn bgra_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut st = 0;
    while st < src_line.len() {
        tgt_line[st  ] = src_line[st+2];
        tgt_line[st+1] = src_line[st+1];
        tgt_line[st+2] = src_line[st  ];
        tgt_line[st+3] = src_line[st+3];
        st += 4;
    }
}

// ------------------------------------------------------------

fn crc32be(data: &[u8]) -> [u8; 4] {
    Crc32::new().put(data).finish_be()
}

struct Crc32 {
    r: u32
}

impl Crc32 {
    fn new() -> Crc32 { Crc32 { r: 0xffff_ffff } }

    fn put<'a>(&'a mut self, bytes: &[u8]) -> &'a mut Crc32 {
        for &byte in bytes {
            let idx = byte ^ (self.r as u8);
            self.r = (self.r >> 8) ^ CRC32_TABLE[idx as usize];
        }
        self
    }

    fn finish_be(&mut self) -> [u8; 4] {
        let result = u32_to_be(self.r ^ 0xffff_ffff);
        self.r = 0xffff_ffff;
        result
    }
}

static CRC32_TABLE: [u32; 256] = [
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

trait IFRead {
    fn read_u8(&mut self) -> io::Result<u8>;
    fn read_exact(&mut self, buf: &mut[u8]) -> io::Result<()>;
}

impl<R: Read> IFRead for R {
    #[inline]
    fn read_u8(&mut self) -> io::Result<u8> {
        let mut buf = [0];
        match self.read(&mut buf) {
            Ok(n) if n == 1 => Ok(buf[0]),
            _               => error("not enough data"),
        }
    }

    fn read_exact(&mut self, buf: &mut[u8]) -> io::Result<()> {
        let mut ready = 0;
        while ready < buf.len() {
            let got = try!(self.read(&mut buf[ready..]));
            if got == 0 {
                return error("not enough data");
            }
            ready += got;
        }
        Ok(())
    }
}

fn u16_from_be(buf: &[u8]) -> u16 {
    (buf[0] as u16) << 8 | buf[1] as u16
}

fn u16_from_le(buf: &[u8]) -> u16 {
    (buf[1] as u16) << 8 | buf[0] as u16
}

fn u16_to_le(x: u16) -> [u8; 2] {
    let buf = [x as u8, (x >> 8) as u8];
    buf
}

fn u32_from_be(buf: &[u8]) -> u32 {
    (buf[0] as u32) << 24 | (buf[1] as u32) << 16 | (buf[2] as u32) << 8 | buf[3] as u32
}

fn u32_to_be(x: u32) -> [u8; 4] {
    let buf = [(x >> 24) as u8, (x >> 16) as u8,
               (x >>  8) as u8, (x)       as u8];
    buf
}

fn u32_from_le(buf: &[u8]) -> u32 {
    (buf[3] as u32) << 24 | (buf[2] as u32) << 16 | (buf[1] as u32) << 8 | buf[0] as u32
}

fn i32_from_le(buf: &[u8]) -> i32 {
    ((buf[3] as u32) << 24 | (buf[2] as u32) << 16 | (buf[1] as u32) << 8 | buf[0] as u32)
        as i32
}

#[inline]
fn copy_memory(src: &[u8], dst: &mut[u8]) {
    if src.len() != dst.len() {
        panic!("src.len() != dst.len()")
    }
    unsafe {
        ptr::copy(src.as_ptr(), dst.as_mut_ptr(), src.len());
    }
}

#[inline]
fn error<T>(msg: &str) -> Result<T, io::Error> {
    Err(io::Error::new(ErrorKind::Other, msg))
}

impl Debug for Image {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "Image {{ w: {:?}, h: {:?}, fmt: {:?}, buf: [{} bytes] }}",
               self.w, self.h, self.fmt, self.buf.len())
    }
}
