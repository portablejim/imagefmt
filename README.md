**Image loading and saving**
* Returned data is always 8-bit (Y/YA/RGB/RGBA/BGR/BGRA)

| Format | Decoder            | Encoder                           |
| ---    | ---                | ---                               |
| png    | 8-bit              | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted | 8-bit non-paletted                |
| jpeg   | baseline           | nope                              |

```Rust
extern crate imagefmt;
use imagefmt::{ColFmt, ColType};

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
    imagefmt::write("out.tga", pic.w, pic.h, pic.fmt, &pic.pixels,
                                                    ColType::Gray)
                                                        .unwrap();

    // get width, height and color type
    let _info = imagefmt::read_info("hiisi.png").unwrap();
}
```
