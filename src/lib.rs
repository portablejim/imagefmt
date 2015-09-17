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
use std::io::{self, Read, Write, BufReader, BufWriter, ErrorKind, Seek, SeekFrom};
use std::path::Path;
use std::fmt::{self, Debug};
use std::cmp::min;
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
    AY,
    RGB,
    RGBA,
    BGR,
    BGRA,
    ARGB,
    ABGR,
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
    let reader = &mut BufReader::new(file);
    read_info_from(reader)
}

/// Like `read_info` but reads from a reader. If it fails, it seeks back to where started.
pub fn read_info_from<R: Read+Seek>(r: &mut R) -> io::Result<Info> {
    let start = try!(r.seek(SeekFrom::Current(0)));
    let s = |r: &mut R| {
        let _ = r.seek(SeekFrom::Start(start));
    };

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
    read_from(reader, req_fmt)
}

/// Like `read` but reads from a reader.
pub fn read_from<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt) -> io::Result<Image> {
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
    let (mut writer, writefunc) = try!(writer_and_writefunc(filepath));
    writefunc(&mut writer, w, h, src_fmt, data, tgt_type, None)
}

/// Writes a region of an image and converts it to requested color type.
pub fn write_region<P>(filepath: P, w: usize, h: usize, src_fmt: ColFmt, data: &[u8],
                                                                   tgt_type: ColType,
                                                                rx: usize, ry: usize,
                                                                rw: usize, rh: usize)
                                                                    -> io::Result<()>
        where P: AsRef<Path>
{
    if rw == 0 || rh == 0 || rx + rw > w || ry + rh > h {
        return error("invalid region");
    }
    let stride = w * src_fmt.bytes_pp();
    let start = ry * stride + rx * src_fmt.bytes_pp();
    let (mut writer, writefunc) = try!(writer_and_writefunc(filepath));
    writefunc(&mut writer, rw, rh, src_fmt, &data[start..], tgt_type, Some(stride))
}

fn writer_and_writefunc<P>(filepath: P) -> io::Result<(BufWriter<File>, WriteFn)>
        where P: AsRef<Path>
{
    let filepath: &Path = filepath.as_ref();
    let writefunc: WriteFn =
        match filepath.extension().and_then(OsStr::to_str) {
            Some("png") if cfg!(feature = "png") => png::write,
            Some("tga") if cfg!(feature = "tga") => tga::write,
            Some("bmp") if cfg!(feature = "bmp") => bmp::write,
            _ => return error("image type not supported for writing"),
        };
    let file = try!(File::create(filepath));
    let writer = BufWriter::new(file);
    Ok((writer, writefunc))
}

type WriteFn =
    fn(&mut BufWriter<File>, usize, usize, ColFmt, &[u8], ColType, Option<usize>)
                                                               -> io::Result<()>;

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
        let mut result = vec![0u8; data.len()];
        copy_memory(data, &mut result[..]);
        return Ok(Image {
            w   : w,
            h   : h,
            fmt : src_fmt,
            buf : result,
        })
    }

    let (convert, c0, c1, c2, c3) =
        try!(converter(src_fmt, tgt_fmt).map_err(|_| "no such converter"));

    let src_linesize = w * src_fmt.bytes_pp();
    let tgt_linesize = w * tgt_fmt.bytes_pp();
    let mut result = vec![0u8; h * tgt_linesize];

    let mut si = 0;
    let mut ti = 0;
    for _j in (0 .. h) {
        convert(&data[si .. si+src_linesize], &mut result[ti .. ti+tgt_linesize],
                c0, c1, c2, c3);
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
            ColFmt::Y                     => ColType::Gray,
            ColFmt::YA | ColFmt::AY       => ColType::GrayAlpha,
            ColFmt::RGB | ColFmt::BGR     => ColType::Color,
            ColFmt::RGBA | ColFmt::BGRA
            | ColFmt::ARGB | ColFmt::ABGR => ColType::ColorAlpha,
            ColFmt::Auto                  => ColType::Auto,
        }
    }

    pub fn has_alpha(self) -> Option<bool> {
        use ColFmt::*;
        match self {
            YA | AY | RGBA | BGRA | ARGB | ABGR => Some(true),
            Y | RGB | BGR => Some(false),
            Auto => None,
        }
    }

    fn bytes_pp(&self) -> usize {
        use self::ColFmt::*;
        match *self {
            Auto                      => 0,
            Y                         => 1,
            YA | AY                   => 2,
            RGB  | BGR                => 3,
            RGBA | BGRA | ARGB | ABGR => 4,
        }
    }

    fn indices_yrgba(self) -> (usize, usize, usize, usize, usize) {
        match self {
            ColFmt::Y    => (0, 0, 0, 0, 0),
            ColFmt::YA   => (0, 0, 0, 0, 1),
            ColFmt::AY   => (1, 0, 0, 0, 0),
            ColFmt::RGB  => (0, 0, 1, 2, 0),
            ColFmt::BGR  => (0, 2, 1, 0, 0),
            ColFmt::RGBA => (0, 0, 1, 2, 3),
            ColFmt::BGRA => (0, 2, 1, 0, 3),
            ColFmt::ARGB => (0, 1, 2, 3, 0),
            ColFmt::ABGR => (0, 3, 2, 1, 0),
            ColFmt::Auto => (0, 0, 0, 0, 0),
        }
    }
}

// ------------------------------------------------------------

type LineConverter = fn(&[u8], &mut[u8], usize, usize, usize, usize);

fn converter(src_fmt: ColFmt, tgt_fmt: ColFmt)
        -> io::Result<(LineConverter, usize, usize, usize, usize)>
{
    use self::ColFmt::*;

    let (syi, sri, sgi, sbi, sai) = src_fmt.indices_yrgba();
    let (tyi, tri, tgi, tbi, tai) = tgt_fmt.indices_yrgba();
    let tci = min(tri, min(tgi, tbi));

    match (src_fmt, tgt_fmt) {
        (Y,Y)|(YA,YA)|(AY,AY)|(RGB,RGB)|(BGR,BGR)
        |(RGBA,RGBA)|(BGRA,BGRA)|(ARGB,ARGB)|(ABGR,ABGR)
                                => Ok((copy_line, 0, 0, 0, 0)),
        (Y, YA) | (Y, AY)       => Ok((y_to_any_ya, tyi, 0, 0, tai)),
        (Y, RGB) | (Y, BGR)     => Ok((y_to_any_rgb, tri, tgi, tbi, 0)),
        (Y, RGBA)
        | (Y, BGRA)
        | (Y, ABGR)
        | (Y, ARGB)             => Ok((y_to_any_rgba, tri, tgi, tbi, tai)),
        (YA, Y) | (AY, Y)       => Ok((any_ya_to_y, syi, 0, 0, 0)),
        (YA, AY) | (AY, YA)     => Ok((ya_to_ay, 0, 0, 0, 0)),
        (YA, RGB)
        | (YA, BGR)
        | (AY, RGB)
        | (AY, BGR)             => Ok((any_ya_to_any_rgb, tri, tgi, tbi, sai)),
        (YA, RGBA)
        | (YA, BGRA)
        | (YA, ARGB)
        | (YA, ABGR)
        | (AY, RGBA)
        | (AY, BGRA)
        | (AY, ARGB)
        | (AY, ABGR)            => Ok((any_ya_to_any_rgba, syi, sai, tci, tai)),
        (RGB, Y) | (BGR, Y)     => Ok((any_rgba_to_y, sri, sgi, sbi, src_fmt.bytes_pp())),
        (RGB, YA) | (BGR, YA)
        | (RGB, AY) | (BGR, AY) => Ok((any_rgb_to_any_ya, sri, sgi, sbi, tai)),
        (RGB, BGR) | (BGR, RGB) => Ok((rgb_to_bgr, 0, 0, 0, 0)),
        (RGB, RGBA)
        | (RGB, BGRA)
        | (RGB, ARGB)
        | (RGB, ABGR)           => Ok((rgb_to_any_rgba, tri, tgi, tbi, tai)),
        (BGR, RGBA)
        | (BGR, BGRA)
        | (BGR, ARGB)
        | (BGR, ABGR)           => Ok((bgr_to_any_rgba, tri, tgi, tbi, tai)),
        (RGBA, Y)
        | (BGRA, Y)
        | (ARGB, Y)
        | (ABGR, Y)             => Ok((any_rgba_to_y, sri, sgi, sbi, src_fmt.bytes_pp())),
        (RGBA, YA)
        | (BGRA, YA)
        | (ARGB, YA)
        | (ABGR, YA)            => Ok((any_rgba_to_ya, sri, sgi, sbi, sai)),
        (RGBA, AY)
        | (BGRA, AY)
        | (ARGB, AY)
        | (ABGR, AY)            => Ok((any_rgba_to_ay, sri, sgi, sbi, sai)),
        (RGBA, RGB)
        | (BGRA, RGB)
        | (ARGB, RGB)
        | (ABGR, RGB)           => Ok((any_rgba_to_rgb, sri, sgi, sbi, sai)),
        (RGBA, BGR)
        | (BGRA, BGR)
        | (ARGB, BGR)
        | (ABGR, BGR)           => Ok((any_rgba_to_bgr, sri, sgi, sbi, sai)),
        (RGBA, BGRA)
        | (RGBA, ARGB)
        | (RGBA, ABGR)          => Ok((rgba_to_any_rgba, tri, tgi, tbi, tai)),
        (BGRA, RGBA)
        | (BGRA, ARGB)
        | (BGRA, ABGR)          => Ok((bgra_to_any_rgba, tri, tgi, tbi, tai)),
        (ARGB, RGBA)
        | (ARGB, BGRA)
        | (ARGB, ABGR)          => Ok((argb_to_any_rgba, tri, tgi, tbi, tai)),
        (ABGR, RGBA)
        | (ABGR, BGRA)
        | (ABGR, ARGB)          => Ok((abgr_to_any_rgba, tri, tgi, tbi, tai)),
        (Auto, _) | (_, Auto) => error("no such converter"),
    }
}

fn copy_line(src: &[u8], tgt: &mut[u8], _: usize, _: usize, _: usize, _: usize) {
    copy_memory(src, tgt)
}

fn y_to_any_ya(src: &[u8], tgt: &mut[u8], yi: usize, _: usize, _: usize, ai: usize) {
    let mut t = 0;
    for &sb in src {
        tgt[t+yi] = sb;
        tgt[t+ai] = 255;
        t += 2;
    }
}

fn y_to_any_rgb(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, _: usize) {
    let mut t = 0;
    for &sb in src {
        tgt[t+ri] = sb;
        tgt[t+gi] = sb;
        tgt[t+bi] = sb;
        t += 3;
    }
}

fn y_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize) {
    let mut t = 0;
    for &sb in src {
        tgt[t+ri] = sb;
        tgt[t+gi] = sb;
        tgt[t+bi] = sb;
        tgt[t+ai] = 255;
        t += 4;
    }
}

fn ya_to_ay(src: &[u8], tgt: &mut[u8], _: usize, _: usize, _: usize, _: usize) {
    let mut st = 0;
    while st < src.len() {
        tgt[st  ] = src[st+1];
        tgt[st+1] = src[st  ];
        st += 2;
    }
}

fn any_ya_to_y(src: &[u8], tgt: &mut[u8], yi: usize, _: usize, _: usize, _: usize) {
    let mut s = 0;
    for tb in tgt {
        *tb = src[s+yi];
        s += 2;
    }
}

fn any_ya_to_any_rgb(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize,
                                                                     sai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+ri] = src[s+(1-sai)];
        tgt[t+gi] = src[s+(1-sai)];
        tgt[t+bi] = src[s+(1-sai)];
        s += 2;
        t += 3;
    }
}

fn any_ya_to_any_rgba(src: &[u8], tgt: &mut[u8], syi: usize, sai: usize, tci: usize,
                                                                         tai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+tci  ] = src[s+syi];
        tgt[t+tci+1] = src[s+syi];
        tgt[t+tci+2] = src[s+syi];
        tgt[t+tai]   = src[s+sai];
        s += 2;
        t += 4;
    }
}

fn any_rgba_to_y(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize,
                                                            bytes_pp: usize)
{
    let mut s = 0;
    for tb in tgt {
        *tb = luminance(src[s+ri], src[s+gi], src[s+bi]);
        s += bytes_pp;
    }
}

fn any_rgb_to_any_ya(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize,
                                                                     tai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+1-tai] = luminance(src[s+ri], src[s+gi], src[s+bi]);
        tgt[t+tai] = 255;
        s += 3;
        t += 2;
    }
}

fn any_rgba_to_ya(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t  ] = luminance(src[s+ri], src[s+gi], src[s+bi]);
        tgt[t+1] = src[s+ai];
        s += 4;
        t += 2;
    }
}

fn any_rgba_to_ay(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+1] = luminance(src[s+ri], src[s+gi], src[s+bi]);
        tgt[t  ] = src[s+ai];
        s += 4;
        t += 2;
    }
}

fn rgb_to_bgr(src: &[u8], tgt: &mut[u8], _: usize, _: usize, _: usize, _: usize)
{
    let mut st = 0;
    while st < src.len() {
        tgt[st  ] = src[st+2];
        tgt[st+1] = src[st+1];
        tgt[st+2] = src[st  ];
        st += 3;
    }
}

fn rgb_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+ri] = src[s  ];
        tgt[t+gi] = src[s+1];
        tgt[t+bi] = src[s+2];
        tgt[t+ai] = 255;
        s += 3;
        t += 4;
    }
}

fn bgr_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+ri] = src[s+2];
        tgt[t+gi] = src[s+1];
        tgt[t+bi] = src[s  ];
        tgt[t+ai] = 255;
        s += 3;
        t += 4;
    }
}

fn any_rgba_to_rgb(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, _ai: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t  ] = src[s+ri];
        tgt[t+1] = src[s+gi];
        tgt[t+2] = src[s+bi];
        s += 4;
        t += 3;
    }
}

fn any_rgba_to_bgr(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, _: usize)
{
    let mut s = 0;
    let mut t = 0;
    while s < src.len() {
        tgt[t+2] = src[s+ri];
        tgt[t+1] = src[s+gi];
        tgt[t  ] = src[s+bi];
        s += 4;
        t += 3;
    }
}

fn rgba_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut st = 0;
    while st < src.len() {
        tgt[st+ri] = src[st+0];
        tgt[st+gi] = src[st+1];
        tgt[st+bi] = src[st+2];
        tgt[st+ai] = src[st+3];
        st += 4;
    }
}

fn bgra_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut st = 0;
    while st < src.len() {
        tgt[st+ri] = src[st+2];
        tgt[st+gi] = src[st+1];
        tgt[st+bi] = src[st+0];
        tgt[st+ai] = src[st+3];
        st += 4;
    }
}

fn argb_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut st = 0;
    while st < src.len() {
        tgt[st+ri] = src[st+1];
        tgt[st+gi] = src[st+2];
        tgt[st+bi] = src[st+3];
        tgt[st+ai] = src[st+0];
        st += 4;
    }
}

fn abgr_to_any_rgba(src: &[u8], tgt: &mut[u8], ri: usize, gi: usize, bi: usize, ai: usize)
{
    let mut st = 0;
    while st < src.len() {
        tgt[st+ri] = src[st+3];
        tgt[st+gi] = src[st+2];
        tgt[st+bi] = src[st+1];
        tgt[st+ai] = src[st+0];
        st += 4;
    }
}

fn luminance(r: u8, g: u8, b: u8) -> u8 {
    (0.21 * r as f32 + 0.64 * g as f32 + 0.15 * b as f32) as u8
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
fn u32_to_le(x: u32) -> [u8; 4] {
    [(x) as u8, (x >> 8) as u8, (x >> 16) as u8, (x >> 24) as u8]
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
                         _: ColType, _: Option<usize>) -> io::Result<()> { panic!("bug") }
        }
    }
}

#[cfg(not(feature = "png"))] dummy_mod!(png);
#[cfg(not(feature = "tga"))] dummy_mod!(tga);
#[cfg(not(feature = "bmp"))] dummy_mod!(bmp);
#[cfg(not(feature = "jpeg"))] dummy_mod!(jpeg);
