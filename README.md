**Image loading and saving**
* Returned data is always 8-bit (Y/YA/RGB/RGBA/BGR/BGRA)
* You can choose to copy just the imagefmt.rs file to your project and avoid all
dependencies completely. Although I think avoiding deps has gone out of fashion.

| Format | Decoder            | Encoder                           |
| ---    | ---                | ---                               |
| png    | 8-bit              | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted | 8-bit non-paletted                |
| jpeg   | baseline           | nope                              |

If you copy the file, the use is as follows:
```Rust
#![feature(core, step_by, rustc_private)]

use imagefmt::{ColFmt, ColType};
mod imagefmt;

fn main() {
    // load and convert to bgra
    let _pic = imagefmt::read("stars.jpg", ColFmt::BGRA).unwrap();

    // convert to grayscale+alpha
    let _pic = imagefmt::read("advanced.png", ColFmt::YA).unwrap();

    // convert to one of y, ya, rgb, rgba
    let pic = imagefmt::read("marbles.tga", ColFmt::Auto).unwrap();

    // write image out as grayscale
    pic.write("out.png", ColType::Gray).unwrap();

    // there's also a free function that doesn't require an Image
    imagefmt::write("out.tga", pic.w, pic.h, &pic.pixels, pic.fmt,
                                                    ColType::Gray)
                                                        .unwrap();

    // get width, height and color type
    let _info = imagefmt::read_info("hiisi.png").unwrap();
}
```
