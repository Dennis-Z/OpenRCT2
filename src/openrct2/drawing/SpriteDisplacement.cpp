#include "SpriteDisplacement.h"
#include <fstream>
#include <unordered_map>
#include <openrct2/core/Memory.hpp>
#include <openrct2/core/Path.hpp>
#include <openrct2/core/String.hpp>
#include <openrct2/core/Math.hpp>

extern "C" {
    #include <openrct2/platform/platform.h>
    #include "drawing.h"
}

static std::unordered_map<SpriteShape, std::string> known_displacements;

rct_drawpixelinfo* get_sprite_dpi(uint32 image) {
    rct_g1_element * g1Element = gfx_get_g1_element(image & 0x7FFFF);
    sint32 width = g1Element->width;
    sint32 height = g1Element->height;

    size_t numPixels = width * height;
    uint8 * pixels8 = Memory::Allocate<uint8>(numPixels);
    Memory::Set(pixels8, 0, numPixels);

    rct_drawpixelinfo* dpi = Memory::Allocate<rct_drawpixelinfo>();
    dpi->bits = pixels8;
    dpi->pitch = 0;
    dpi->x = 0;
    dpi->y = 0;
    dpi->width = width;
    dpi->height = height;
    dpi->zoom_level = 0;

    gfx_draw_sprite_software(dpi, image, -g1Element->x_offset, -g1Element->y_offset, 0);

    return dpi;
}

void free_sprite_dpi(rct_drawpixelinfo* dpi) {
    Memory::Free(dpi->bits);
    Memory::Free(dpi);
}

SpriteShape get_sprite_shape(uint32 image) {
    rct_drawpixelinfo* dpi = get_sprite_dpi(image);

    std::bitset<64 * 64> shapeMap;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            if (y < dpi->height && x < dpi->width)
                shapeMap[y * 64 + x] = dpi->bits[y * dpi->width + x] ? 1 : 0;
            else
                shapeMap[y * 64 + x] = 0;
        }

    free_sprite_dpi(dpi);

    return shapeMap;
}

SpriteDisplacement estimate_sprite_displacement(uint32 image);
SpriteDisplacement load_sprite_displacement_from_file(std::string fname);

SpriteDisplacement get_sprite_displacement(uint32 image) {
    SpriteShape shape = get_sprite_shape(image);
    auto known_displacement = known_displacements.find(shape);
    if (known_displacement != known_displacements.end()) {
        return load_sprite_displacement_from_file(known_displacement->second);
    }
    return estimate_sprite_displacement(image);
}

int load_known_displacements() {
    if (known_displacements.empty()) {
        char buffer[512];
        platform_get_openrct_data_path(buffer, 512);
        Path::Append(buffer, 512, "displacements");
        Path::Append(buffer, 512, "list");

        std::ifstream stream(buffer, std::ios::in);
        if (stream.fail()) return 1;

        // TODO: this format is really rough, should probably redo this
        // the list file is formatted like:
        // displacementname 01001010101010101010101110101010101010101010 (the shape)
        // .... 0101010
        // ## comment ##

        bool is_comment = false;
        std::string token;
        std::string reading_fname = "";

        while (!stream.eof()) {
            stream >> token;
            if (token.length() > 0) {
                if (token == "##") is_comment = !is_comment;
                else if (!is_comment) {
                    if (reading_fname.length() == 0) reading_fname = token;
                    else {
                        if (token.length() != SPRITESHAPE_SIZE_X * SPRITESHAPE_SIZE_Y) {
                            stream.close();
                            return 1;
                        }
                        known_displacements[SpriteShape(token)] = reading_fname;
                        reading_fname = "";
                    }
                }
            }
        }

        stream.close();

        log_info("Loaded %d displacement mappings", known_displacements.size());
    }

    return 0;
}

int link_shape_to_displacement_file(uint32 image, const char* fname) {
    // appends a line to the displacement file containing the specified link

    char buffer[512];
    platform_get_openrct_data_path(buffer, 512);
    Path::Append(buffer, 512, "displacements");
    Path::Append(buffer, 512, "list");

    std::ofstream stream(buffer, std::ios::out | std::ios::app);
    if (stream.fail()) return 1;

    stream << fname << " ";
    stream << get_sprite_shape(image).to_string() << "\n";

    stream.close();

    return 0;
}

SpriteDisplacement load_sprite_displacement_from_file(std::string fname) {
    char buffer[512];
    platform_get_openrct_data_path(buffer, 512);
    Path::Append(buffer, 512, "displacements");
    Path::Append(buffer, 512, fname.c_str());
    String::Append(buffer, 512, ".raw");

    std::ifstream stream(buffer, std::ios::in | std::ios::binary);
    if (stream.fail()) throw std::runtime_error("Cannot read file " + std::string(buffer));

    std::vector<uint8> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    stream.close();

    // displacement files are formatted like this:
    // [uint8 width] [uint8 height] ([uint8 xdisplacement] [uint8 ydisplacement] [uint8 zdisplacement]){width*height}
    // so total size is ((width*height*3) + 2) bytes

    if (contents.size() < 2) throw std::runtime_error("File " + std::string(fname) + " is too small to contain displacement data");

    SpriteDisplacement displacement(contents[0], contents[1]);

    if (contents.size() != displacement.width * displacement.height * 3u + 2u) throw std::runtime_error("File " + std::string(fname) + " is not properly formatted for displacement data");

    memcpy(displacement.data.data(), contents.data() + 2, displacement.width * displacement.height * 3);

    return displacement;
}

SpriteDisplacement estimate_sprite_displacement(uint32 image) {
    rct_drawpixelinfo* dpi = get_sprite_dpi(image);

    SpriteDisplacement displacement(dpi->width, dpi->height);

    int leftBottomOffY = 100000;
    int leftBottomX = 0;
    int leftBottomY = 0;
    int rightBottomOffY = 100000;
    int rightBottomX = 0;
    int rightBottomY = 0;
    int leftTopOffY = -100000;
    int leftTopX = 0;
    int leftTopY = 0;
    int rightTopOffY = -100000;
    int rightTopX = 0;
    int rightTopY = 0;
    int leftX = displacement.width - 1;
    int rightX = 0;

    // find the pixels on the left/right top/bottom "edges" of the bounding box
    //    /T\
    //  LT   RT
    //  /     \
    // |\     /|
    // | \   / |
    //  \ \ / /
    //  LB v RB
    //    \B/
    for (int y = displacement.height - 1; y >= 0; y--) {
        for (int x = 0; x < displacement.width; x++) {
            if (dpi->bits[y * displacement.width + x]) {
                int thisLeftOffY = (x) - y * 2;
                int thisRightOffY = (dpi->width - x) - y * 2;
                if (thisLeftOffY < leftBottomOffY) {
                    leftBottomOffY = thisLeftOffY;
                    leftBottomX = x;
                    leftBottomY = y;
                }
                if (thisRightOffY < rightBottomOffY) {
                    rightBottomOffY = thisRightOffY;
                    rightBottomX = x;
                    rightBottomY = y;
                }
                if (thisRightOffY > leftTopOffY) {
                    leftTopOffY = thisRightOffY;
                    leftTopX = x;
                    leftTopY = y;
                }
                if (thisLeftOffY > rightTopOffY) {
                    rightTopOffY = thisLeftOffY;
                    rightTopX = x;
                    rightTopY = y;
                }

                if (x < leftX) leftX = x;
                if (x > rightX) rightX = x;
            }
        }
    }

    // from the found pixels, determine top and bottom
    int topX, topY, bottomX, bottomY;

    // correct right side so that Y values equal
    int topDeltaY = rightTopY - leftTopY;
    rightTopX -= topDeltaY * 2;
    rightTopY -= topDeltaY;

    int bottomDeltaY = rightBottomY - leftBottomY;
    rightBottomX -= bottomDeltaY * 2;
    rightBottomY -= bottomDeltaY;

    topX = (rightTopX + leftTopX) / 2;
    bottomX = (rightBottomX + leftBottomX) / 2;
    topY = rightTopY - (rightTopX - leftTopX) / 4; // half difference and divide by 2
    bottomY = rightBottomY + (rightBottomX - leftBottomX) / 4;

    int leftWallYTop = topY + (topX - leftX) / 2;
    int leftWallYBottom = bottomY - (bottomX - leftX) / 2;
    int rightWallYTop = topY + (rightX - topX) / 2;
    int rightWallYBottom = bottomY - (rightX - bottomX) / 2;
    int wallHeight = Math::Min(leftWallYBottom - leftWallYTop, rightWallYBottom - rightWallYTop);
    if (wallHeight < 8) wallHeight = 0;

    int wallHeight2 = wallHeight * 2;

    for (int y = dpi->height - 1; y >= 0; y--) {
        for (int x = 0; x < dpi->width; x++) {
            int dx = bottomX - x;
            int dy = ((dpi->height - 1) - y) * 2 - abs(dx);
            off_t offset = dpi->width * 3 * y + 3 * x;
            displacement.data[offset + 0] = Math::Clamp(0, -dx * 2, 255);
            displacement.data[offset + 1] = Math::Clamp(0, dx * 2, 255);
            displacement.data[offset + 2] = Math::Clamp(0, dy / 2, Math::Min(255, wallHeight));
            if (dy > wallHeight2) {
                displacement.data[offset + 0] = Math::Clamp(0, (int)displacement.data[offset + 0] + (dy - wallHeight2), 255);
                displacement.data[offset + 1] = Math::Clamp(0, (int)displacement.data[offset + 1] + (dy - wallHeight2), 255);
            }
        }
    }

    free_sprite_dpi(dpi);

    return displacement;
}