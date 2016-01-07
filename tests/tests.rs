extern crate imagefmt;
use imagefmt::{Image, ColFmt};

fn equal(pics: &[&Image<u8>]) -> bool {
    if pics.len() < 2 { assert!(false) }
    let a = pics[0];
    for &b in &pics[1..] {
        if a.w != b.w || a.h != b.h || a.fmt != b.fmt || a.buf != b.buf {
            return false
        }
    }
    true
}

#[test]
fn png_tga_bmp() {
    // The TGA and BMP files are not as varied in format as the PNG files, so
    // not as well tested.
    let png_path = "tests/pngsuite/";
    let tga_path = "tests/pngsuite-tga/";
    let bmp_path = "tests/pngsuite-bmp/";

    let names = [
        "basi0g08",    // PNG image data, 32 x 32, 8-bit grayscale, interlaced
        "basi2c08",    // PNG image data, 32 x 32, 8-bit/color RGB, interlaced
        "basi3p08",    // PNG image data, 32 x 32, 8-bit colormap, interlaced
        "basi4a08",    // PNG image data, 32 x 32, 8-bit gray+alpha, interlaced
        "basi6a08",    // PNG image data, 32 x 32, 8-bit/color RGBA, interlaced
        "basn0g08",    // PNG image data, 32 x 32, 8-bit grayscale, non-interlaced
        "basn2c08",    // PNG image data, 32 x 32, 8-bit/color RGB, non-interlaced
        "basn3p08",    // PNG image data, 32 x 32, 8-bit colormap, non-interlaced
        "basn4a08",    // PNG image data, 32 x 32, 8-bit gray+alpha, non-interlaced
        "basn6a08",    // PNG image data, 32 x 32, 8-bit/color RGBA, non-interlaced
    ];

    for name in &names {
        let a = imagefmt::read(&format!("{}{}.png", png_path, name), ColFmt::RGBA).unwrap();
        let b = imagefmt::read(&format!("{}{}.tga", tga_path, name), ColFmt::RGBA).unwrap();
        let c = imagefmt::read(&format!("{}{}.bmp", bmp_path, name), ColFmt::RGBA).unwrap();
        assert!(equal(&[&a, &b, &c]));
    }
}

#[test]
fn conversions() {
    // Not exhaustive...

    let a = imagefmt::read("tests/pngsuite/basn6a08.png", ColFmt::RGBA).unwrap();
    let b = imagefmt::convert(a.w, a.h, a.fmt, &a.buf, ColFmt::BGRA).unwrap();
    let b = imagefmt::convert(b.w, b.h, b.fmt, &b.buf, ColFmt::ARGB).unwrap();
    let b = imagefmt::convert(b.w, b.h, b.fmt, &b.buf, ColFmt::ABGR).unwrap();
    let b = imagefmt::convert(b.w, b.h, b.fmt, &b.buf, ColFmt::RGBA).unwrap();
    assert!(equal(&[&a, &b]));
    let c = imagefmt::convert(a.w, a.h, a.fmt, &a.buf, ColFmt::RGB).unwrap();
    let d = imagefmt::convert(a.w, a.h, a.fmt, &a.buf, ColFmt::BGR).unwrap();
    let e = imagefmt::convert(d.w, d.h, d.fmt, &d.buf, ColFmt::RGB).unwrap();
    assert!(equal(&[&c, &e]));
    let ay = imagefmt::convert(a.w, a.h, a.fmt, &a.buf, ColFmt::AY).unwrap();
    let ya = imagefmt::convert(b.w, b.h, b.fmt, &b.buf, ColFmt::YA).unwrap();
    let b = imagefmt::convert(ya.w, ya.h, ya.fmt, &ya.buf, ColFmt::AY).unwrap();
    assert!(equal(&[&ay, &b]));
    let b = imagefmt::convert(b.w, b.h, b.fmt, &b.buf, ColFmt::Y).unwrap();
    let c = imagefmt::convert(c.w, c.h, c.fmt, &c.buf, ColFmt::Y).unwrap();
    let d = imagefmt::convert(d.w, d.h, d.fmt, &d.buf, ColFmt::Y).unwrap();
    let g = imagefmt::convert(a.w, a.h, a.fmt, &a.buf, ColFmt::Y).unwrap();
    assert!(equal(&[&b, &c, &d, &g]));

    // From gray to color+alpha.
    let rgba = imagefmt::convert(g.w, g.h, g.fmt, &g.buf, ColFmt::RGBA).unwrap();
    for &fmt in &[ColFmt::BGRA, ColFmt::ARGB, ColFmt::ABGR] {
        let ca = imagefmt::convert(g.w, g.h, g.fmt, &g.buf, fmt).unwrap();
        let ca = imagefmt::convert(ca.w, ca.h, ca.fmt, &ca.buf, ColFmt::RGBA).unwrap();
        assert!(equal(&[&ca, &rgba]));
    }

    // From gray+alpha to color+alpha.
    let rgba0 = imagefmt::convert(ya.w, ya.h, ya.fmt, &ya.buf, ColFmt::RGBA).unwrap();
    let rgba1 = imagefmt::convert(ay.w, ay.h, ay.fmt, &ay.buf, ColFmt::RGBA).unwrap();
    for &fmt in &[ColFmt::BGRA, ColFmt::ARGB, ColFmt::ABGR] {
        let ca0 = imagefmt::convert(ya.w, ya.h, ya.fmt, &ya.buf, fmt).unwrap();
        let ca0 = imagefmt::convert(ca0.w, ca0.h, ca0.fmt, &ca0.buf, ColFmt::RGBA).unwrap();
        let ca1 = imagefmt::convert(ay.w, ay.h, ay.fmt, &ay.buf, fmt).unwrap();
        let ca1 = imagefmt::convert(ca1.w, ca1.h, ca1.fmt, &ca1.buf, ColFmt::RGBA).unwrap();
        assert!(equal(&[&ca0, &ca1, &rgba0, &rgba1]));
    }
}
