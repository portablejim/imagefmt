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
use std::io::{self, Read, BufReader, BufWriter, ErrorKind, Seek, SeekFrom};
use std::iter::{repeat};
use std::path::Path;
use std::fmt::{self, Debug};
use std::ptr;

#[cfg(feature = "png")] pub mod png;
#[cfg(feature = "tga")] pub mod tga;
#[cfg(feature = "bmp")] pub mod bmp;
#[cfg(feature = "jpeg")] pub mod jpeg;

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
    let file = try!(File::open(filepath));
    let r = &mut BufReader::new(file);

    fn s(r: &mut BufReader<File>) {
        let _ = r.seek(SeekFrom::Start(0));
    }

    if cfg!(feature = "png") { match png::read_info(r) { Ok(i) => return Ok(i), _ => {s(r)} } }
    if cfg!(feature = "jpeg") { match jpeg::read_info(r) { Ok(i) => return Ok(i), _ => {s(r)} } }
    if cfg!(feature = "bmp") { match bmp::read_info(r) { Ok(i) => return Ok(i), _ => {s(r)} } }
    if cfg!(feature = "tga") { match tga::read_info(r) { Ok(i) => return Ok(i), _ => {s(r)} } }
    error("image type not recognized")
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to one of `Y`, `YA`, `RGB`,
/// `RGBA`.
pub fn read<P: AsRef<Path>>(filepath: P, req_fmt: ColFmt) -> io::Result<Image> {
    let file = try!(File::open(filepath));
    let reader = &mut BufReader::new(file);
    read_from_reader(reader, req_fmt)
}

/// Like `read` but reads from a reader.
pub fn read_from_reader<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt)
                                                  -> io::Result<Image>
{
    if      cfg!(feature = "png") && png::detect(reader) { png::read(reader, req_fmt) }
    else if cfg!(feature = "jpeg") && jpeg::detect(reader) { jpeg::read(reader, req_fmt) }
    else if cfg!(feature = "bmp") && bmp::detect(reader) { bmp::read(reader, req_fmt) }
    else if cfg!(feature = "tga") && tga::detect(reader) { tga::read(reader, req_fmt) }
    else { error("image type not recognized") }
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
            Some("png") if cfg!(feature = "png") => png::write,
            Some("tga") if cfg!(feature = "tga") => tga::write,
            _ => return error("image type not supported for writing"),
        };
    let file = try!(File::create(filepath));
    let writer = &mut BufWriter::new(file);
    writefunc(writer, w, h, src_fmt, data, tgt_type)
}

/// Converts the image into a new allocation.
pub fn convert(w: usize, h: usize, src_fmt: ColFmt, data: &[u8], tgt_fmt: ColFmt)
                                                           -> Result<Image, &str>
{
    if w < 1 || h < 1 || src_fmt.bytes_pp() * w * h != data.len() {
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

    pub fn has_alpha(self) -> Option<bool> {
        use ColFmt::*;
        match self {
            YA | RGBA | BGRA => Some(true),
            Y | RGB | BGR => Some(false),
            Auto => None,
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

trait IFRead {
    fn read_u8(&mut self) -> io::Result<u8>;
    fn read_exact_(&mut self, buf: &mut[u8]) -> io::Result<()>;
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

    fn read_exact_(&mut self, buf: &mut[u8]) -> io::Result<()> {
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

#[allow(dead_code)]
fn u16_from_be(buf: &[u8]) -> u16 {
    (buf[0] as u16) << 8 | buf[1] as u16
}

#[allow(dead_code)]
fn u16_from_le(buf: &[u8]) -> u16 {
    (buf[1] as u16) << 8 | buf[0] as u16
}

#[allow(dead_code)]
fn u16_to_le(x: u16) -> [u8; 2] {
    [x as u8, (x >> 8) as u8]
}

#[allow(dead_code)]
fn u32_from_be(buf: &[u8]) -> u32 {
    (buf[0] as u32) << 24 | (buf[1] as u32) << 16 | (buf[2] as u32) << 8 | buf[3] as u32
}

#[allow(dead_code)]
fn u32_to_be(x: u32) -> [u8; 4] {
    [(x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, (x) as u8]
}

#[allow(dead_code)]
fn u32_from_le(buf: &[u8]) -> u32 {
    (buf[3] as u32) << 24 | (buf[2] as u32) << 16 | (buf[1] as u32) << 8 | buf[0] as u32
}

#[allow(dead_code)]
fn i32_from_le(buf: &[u8]) -> i32 {
    ((buf[3] as u32) << 24 | (buf[2] as u32) << 16 | (buf[1] as u32) << 8 | buf[0] as u32)
        as i32
}

#[inline]
fn copy_memory(src: &[u8], dst: &mut[u8]) {
    assert!(src.len() == dst.len());
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

macro_rules! dummy_mod {
    ($name:ident) => {
        #[allow(dead_code)]
        mod $name {
            use ::{Info, Image, ColFmt, ColType};
            use std::io::{self, Read, Write};
            pub fn detect<R: Read>(_: &mut R) -> bool { panic!("bug") }
            pub fn read_info<R: Read>(_: &mut R) -> io::Result<Info> { panic!("bug") }
            pub fn read<R: Read>(_: &mut R, _: ColFmt) -> io::Result<Image> { panic!("bug") }
            pub fn write<W: Write>(_: &mut W, _: usize, _: usize, _: ColFmt, _: &[u8],
                                           _: ColType) -> io::Result<()> { panic!("bug") }
        }
    }
}

#[cfg(not(feature = "png"))] dummy_mod!(png);
#[cfg(not(feature = "tga"))] dummy_mod!(tga);
#[cfg(not(feature = "bmp"))] dummy_mod!(bmp);
#[cfg(not(feature = "jpeg"))] dummy_mod!(jpeg);
