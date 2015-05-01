**Image loading and saving**
* Returned data is always 8-bit (Y/YA/RGB/RGBA)

| Format | Decoder            | Encoder                           |
| ---    | ---                | ---                               |
| png    | 8-bit              | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted | 8-bit non-paletted                |
| jpeg   | baseline           | nope                              |

```Rust
#![feature(core)]
#![feature(step_by)]
#![feature(rustc_private)]

use std::path::Path;

use imageformats::*;
mod imageformats;

fn main() {
    // load and convert to rgba
    let _pic = read_image(&Path::new("stars.jpg"), ColFmt::RGBA).unwrap();

    // convert to grayscale+alpha
    let _pic = read_image(&Path::new("advanced.png"), ColFmt::YA).unwrap();

    // no conversion
    let pic = read_image(&Path::new("marbles.tga"), ColFmt::Auto).unwrap();

    // write image out as grayscale
    write_image(&Path::new("out.png"), pic.w, pic.h, &pic.pixels, ColFmt::Y).unwrap();

    // print width, heigth and color format
    println!("{:?}", read_image_info(&Path::new("hiisi.png")).unwrap());
}
```
