**Image loading and saving**
* Returned data is always 8-bit (Y/YA/RGB/RGBA)

| Format | Decoder            | Encoder                           |
| ---    | ---                | ---                               |
| png    | 8-bit              | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted | 8-bit non-paletted                |
| jpeg   | baseline           | nope                              |

```Rust
#![feature(core, step_by, rustc_private)]

use imagefmt::ColFmt;
mod imagefmt;

fn main() {
    // load and convert to rgba
    let _pic = imagefmt::read("stars.jpg", ColFmt::RGBA).unwrap();

    // convert to grayscale+alpha
    let _pic = imagefmt::read("advanced.png", ColFmt::YA).unwrap();

    // no conversion
    let pic = imagefmt::read("marbles.tga", ColFmt::Auto).unwrap();

    // write image out as grayscale
    imagefmt::write("out.png", pic.w, pic.h, &pic.pixels, ColFmt::Y).unwrap();

    // print width, heigth and color format
    println!("{:?}", imagefmt::read_info("hiisi.png").unwrap());
}
```
