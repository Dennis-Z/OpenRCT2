#pragma region Copyright (c) 2014-2017 OpenRCT2 Developers
/*****************************************************************************
 * OpenRCT2, an open source clone of Roller Coaster Tycoon 2.
 *
 * OpenRCT2 is the work of many authors, a full list can be found in contributors.md
 * For more information, visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * A full copy of the GNU General Public License can be found in licence.txt
 *****************************************************************************/
#pragma endregion

#ifndef DISABLE_OPENGL

#include <fstream>
#include <vector>
#include <stdexcept>
#include <openrct2/core/Memory.hpp>
#include <openrct2/core/Path.hpp>
#include <openrct2/core/String.hpp>
#include "TextureCache.h"
#include "openrct2/core/Math.hpp"

extern "C"
{
    #include <openrct2/drawing/drawing.h>
    #include <openrct2/platform/platform.h>
}

TextureCache::~TextureCache()
{
    FreeTextures();
}

void TextureCache::InvalidateImage(uint32 image)
{
    auto kvp = _imageTextureMap.find(image);
    if (kvp != _imageTextureMap.end())
    {
        _atlases[kvp->second.index].Free(kvp->second);
        _imageTextureMap.erase(kvp);
    }
}

void TextureCache::CreateAtlasesTexture()
{
    if (!_atlasesTextureInitialised)
    {
        // Determine width and height to use for texture atlases
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &_atlasesTextureDimensions);
        if (_atlasesTextureDimensions > TEXTURE_CACHE_MAX_ATLAS_SIZE) {
            _atlasesTextureDimensions = TEXTURE_CACHE_MAX_ATLAS_SIZE;
        }

        // Determine maximum number of atlases (minimum of size and array limit)
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &_atlasesTextureIndicesLimit);
        if (_atlasesTextureDimensions < _atlasesTextureIndicesLimit) _atlasesTextureIndicesLimit = _atlasesTextureDimensions;

        AllocateAtlasesTexture();

        _atlasesTextureInitialised = true;
        _atlasesTextureIndices = 0;
    }
}

void TextureCache::AllocateAtlasesTexture()
{
    // Create an array texture to hold all of the atlases
    glGenTextures(1, &_atlasesTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

CachedTextureInfo TextureCache::AllocateImage(sint32 imageWidth, sint32 imageHeight)
{
    CreateAtlasesTexture();

    // Find an atlas that fits this image
    for (Atlas& atlas : _atlases)
    {
        if (atlas.GetFreeSlots() > 0 && atlas.IsImageSuitable(imageWidth, imageHeight))
        {
            return atlas.Allocate(imageWidth, imageHeight);
        }
    }

    // If there is no such atlas, then create a new one
    if ((sint32)_atlases.size() >= _atlasesTextureIndicesLimit)
    {
        throw std::runtime_error("more texture atlases required, but device limit reached!");
    }

    sint32 atlasIndex = (sint32)_atlases.size();
    sint32 atlasSize = (sint32)powf(2, (float)Atlas::CalculateImageSizeOrder(imageWidth, imageHeight));

#ifdef DEBUG
    log_verbose("new texture atlas #%d (size %d) allocated\n", atlasIndex, atlasSize);
#endif

    _atlases.emplace_back(atlasIndex, atlasSize);
    _atlases.back().Initialise(_atlasesTextureDimensions, _atlasesTextureDimensions);

    // Enlarge texture array to support new atlas
    EnlargeAtlasesTexture(1);

    // And allocate from the new atlas
    return _atlases.back().Allocate(imageWidth, imageHeight);
}

void TextureCache::FreeTextures()
{
    // Free array texture
    glDeleteTextures(1, &_atlasesTexture);
}

rct_drawpixelinfo * TextureCache::CreateDPI(sint32 width, sint32 height)
{
    size_t numPixels = width * height;
    uint8 * pixels8 = Memory::Allocate<uint8>(numPixels);
    Memory::Set(pixels8, 0, numPixels);

    rct_drawpixelinfo * dpi = new rct_drawpixelinfo();
    dpi->bits = pixels8;
    dpi->pitch = 0;
    dpi->x = 0;
    dpi->y = 0;
    dpi->width = width;
    dpi->height = height;
    dpi->zoom_level = 0;
    return dpi;
}

void TextureCache::DeleteDPI(rct_drawpixelinfo* dpi)
{
    Memory::Free(dpi->bits);
    delete dpi;
}

GLuint TextureCache::GetAtlasesTexture()
{
    return _atlasesTexture;
}



void PaletteTextureCache::SetPalette(const SDL_Color * palette)
{
    Memory::CopyArray(_palette, palette, 256);
}

CachedTextureInfo PaletteTextureCache::GetOrLoadImageTexture(uint32 image)
{
    image &= 0x7FFFF;
    auto kvp = _imageTextureMap.find(image);
    if (kvp != _imageTextureMap.end())
    {
        return kvp->second;
    }

    auto cacheInfo = LoadImageTexture(image);
    _imageTextureMap[image] = cacheInfo;

    return cacheInfo;
}

CachedTextureInfo PaletteTextureCache::GetOrLoadPaletteTexture(uint32 image, uint32 tertiaryColour, bool special)
{
    if ((image & (IMAGE_TYPE_REMAP | IMAGE_TYPE_REMAP_2_PLUS | IMAGE_TYPE_TRANSPARENT)) == 0)
        return CachedTextureInfo{ 0 };

    uint32 uniquePaletteId = image & (IMAGE_TYPE_REMAP | IMAGE_TYPE_REMAP_2_PLUS | IMAGE_TYPE_TRANSPARENT);
    if (!(image & IMAGE_TYPE_REMAP_2_PLUS)) {
        uniquePaletteId = (image >> 19) & 0xFF;
        if (!(image & IMAGE_TYPE_TRANSPARENT)) {
            uniquePaletteId &= 0x7F;
        }
    }
    else {
        uniquePaletteId |= ((image >> 19) & 0x1F);
        uniquePaletteId |= ((image >> 24) & 0x1F) << 8;

        if (!(image & IMAGE_TYPE_REMAP)) {
            uniquePaletteId |= tertiaryColour << 16;
        }
    }

    auto kvp = _paletteTextureMap.find(uniquePaletteId);
    if (kvp != _paletteTextureMap.end())
    {
        return kvp->second;
    }

    auto cacheInfo = LoadPaletteTexture(image, tertiaryColour, special);
    _paletteTextureMap[uniquePaletteId] = cacheInfo;

    return cacheInfo;
}

CachedTextureInfo PaletteTextureCache::GetOrLoadGlyphTexture(uint32 image, uint8 * palette)
{
    GlyphId glyphId;
    glyphId.Image = image;
    Memory::Copy<void>(&glyphId.Palette, palette, sizeof(glyphId.Palette));

    auto kvp = _glyphTextureMap.find(glyphId);
    if (kvp != _glyphTextureMap.end())
    {
        return kvp->second;
    }

    auto cacheInfo = LoadGlyphTexture(image, palette);
    _glyphTextureMap[glyphId] = cacheInfo;

    return cacheInfo;
}

void PaletteTextureCache::EnlargeAtlasesTexture(GLuint newEntries)
{
    CreateAtlasesTexture();

    GLuint newIndices = _atlasesTextureIndices + newEntries;

    // Retrieve current array data
    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    auto oldPixels = std::vector<char>(_atlasesTextureDimensions * _atlasesTextureDimensions * _atlasesTextureIndices);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, oldPixels.data());

    // Delete old texture, allocate a new one, then define the new format on the newly created texture
    glDeleteTextures(1, &_atlasesTexture);
    AllocateAtlasesTexture();
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8UI, _atlasesTextureDimensions, _atlasesTextureDimensions, newIndices, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);

    // Restore old data
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, _atlasesTextureDimensions, _atlasesTextureDimensions, _atlasesTextureIndices, GL_RED_INTEGER, GL_UNSIGNED_BYTE, oldPixels.data());

    _atlasesTextureIndices = newIndices;
}

CachedTextureInfo PaletteTextureCache::LoadImageTexture(uint32 image)
{
    rct_drawpixelinfo * dpi = GetImageAsDPI(image, 0);

    auto cacheInfo = AllocateImage(dpi->width, dpi->height);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, cacheInfo.bounds.x, cacheInfo.bounds.y, cacheInfo.index, dpi->width, dpi->height, 1, GL_RED_INTEGER, GL_UNSIGNED_BYTE, dpi->bits);

    DeleteDPI(dpi);

    return cacheInfo;
}

CachedTextureInfo PaletteTextureCache::LoadPaletteTexture(uint32 image, uint32 tertiaryColour, bool special)
{
    rct_drawpixelinfo dpi;
    dpi.bits = gfx_draw_sprite_get_palette(image, tertiaryColour);
    dpi.width = 256;
    dpi.height = special ? 5 : 1;
    auto cacheInfo = AllocateImage(dpi.width, dpi.height);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, cacheInfo.bounds.x, cacheInfo.bounds.y, cacheInfo.index, dpi.width, dpi.height, 1, GL_RED_INTEGER, GL_UNSIGNED_BYTE, dpi.bits);

    return cacheInfo;
}

CachedTextureInfo PaletteTextureCache::LoadGlyphTexture(uint32 image, uint8 * palette)
{
    rct_drawpixelinfo * dpi = GetGlyphAsDPI(image, palette);

    auto cacheInfo = AllocateImage(dpi->width, dpi->height);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, cacheInfo.bounds.x, cacheInfo.bounds.y, cacheInfo.index, dpi->width, dpi->height, 1, GL_RED_INTEGER, GL_UNSIGNED_BYTE, dpi->bits);

    DeleteDPI(dpi);

    return cacheInfo;
}

void * PaletteTextureCache::GetImageAsARGB(uint32 image, uint32 tertiaryColour, uint32 * outWidth, uint32 * outHeight)
{
    rct_g1_element * g1Element = gfx_get_g1_element(image & 0x7FFFF);
    sint32 width = g1Element->width;
    sint32 height = g1Element->height;

    rct_drawpixelinfo * dpi = CreateDPI(width, height);
    gfx_draw_sprite_software(dpi, image, -g1Element->x_offset, -g1Element->y_offset, tertiaryColour);
    void * pixels32 = ConvertDPIto32bpp(dpi);
    DeleteDPI(dpi);

    *outWidth = width;
    *outHeight = height;
    return pixels32;
}

rct_drawpixelinfo * TextureCache::GetImageAsDPI(uint32 image, uint32 tertiaryColour)
{
    rct_g1_element * g1Element = gfx_get_g1_element(image & 0x7FFFF);
    sint32 width = g1Element->width;
    sint32 height = g1Element->height;

    rct_drawpixelinfo * dpi = CreateDPI(width, height);
    gfx_draw_sprite_software(dpi, image, -g1Element->x_offset, -g1Element->y_offset, tertiaryColour);
    return dpi;
}

void * PaletteTextureCache::GetGlyphAsARGB(uint32 image, uint8 * palette, uint32 * outWidth, uint32 * outHeight)
{
    rct_g1_element * g1Element = gfx_get_g1_element(image & 0x7FFFF);
    sint32 width = g1Element->width;
    sint32 height = g1Element->height;

    rct_drawpixelinfo * dpi = CreateDPI(width, height);
    gfx_draw_sprite_palette_set_software(dpi, image, -g1Element->x_offset, -g1Element->y_offset, palette, nullptr);
    void * pixels32 = ConvertDPIto32bpp(dpi);
    DeleteDPI(dpi);

    *outWidth = width;
    *outHeight = height;
    return pixels32;
}

rct_drawpixelinfo * PaletteTextureCache::GetGlyphAsDPI(uint32 image, uint8 * palette)
{
    rct_g1_element * g1Element = gfx_get_g1_element(image & 0x7FFFF);
    sint32 width = g1Element->width;
    sint32 height = g1Element->height;

    rct_drawpixelinfo * dpi = CreateDPI(width, height);
    gfx_draw_sprite_palette_set_software(dpi, image, -g1Element->x_offset, -g1Element->y_offset, palette, nullptr);
    return dpi;
}

void * PaletteTextureCache::ConvertDPIto32bpp(const rct_drawpixelinfo * dpi)
{
    size_t numPixels = dpi->width * dpi->height;
    uint8 * pixels32 = Memory::Allocate<uint8>(numPixels * 4);
    uint8 * src = dpi->bits;
    uint8 * dst = pixels32;
    for (size_t i = 0; i < numPixels; i++)
    {
        uint8 paletteIndex = *src++;
        if (paletteIndex == 0)
        {
            // Transparent
            *dst++ = 0;
            *dst++ = 0;
            *dst++ = 0;
            *dst++ = 0;
        }
        else
        {
            SDL_Color colour = _palette[paletteIndex];
            *dst++ = colour.r;
            *dst++ = colour.g;
            *dst++ = colour.b;
            *dst++ = colour.a;
        }
    }
    return pixels32;
}

CachedTextureInfo DisplacementTextureCache::GetOrLoadDisplacementTexture(uint32 image)
{
    image &= 0x7FFFF;
    auto kvp = _imageTextureMap.find(image);
    if (kvp != _imageTextureMap.end())
    {
        return kvp->second;
    }

    auto cacheInfo = LoadDisplacementTexture(image);
    _imageTextureMap[image] = cacheInfo;

    return cacheInfo;
}

void DisplacementTextureCache::LoadDisplacementsForKnownShapes()
{
    shapeToDisplacement[GetShape(1921)] = "s1lt";
    shapeToDisplacement[GetShape(1918)] = "s1rt";
    shapeToDisplacement[GetShape(1943)] = "s1rb";
    shapeToDisplacement[GetShape(1946)] = "s1lb";
    shapeToDisplacement[GetShape(1942)] = "s2d2";
    shapeToDisplacement[GetShape(1949)] = "s2d";
    shapeToDisplacement[GetShape(1932)] = "s2r";
    shapeToDisplacement[GetShape(1933)] = "s2l";
    shapeToDisplacement[GetShape(1945)] = "s3r";
    shapeToDisplacement[GetShape(1941)] = "s3t";
    shapeToDisplacement[GetShape(1947)] = "s3b";
    shapeToDisplacement[GetShape(1948)] = "s3l";
}

CachedTextureInfo DisplacementTextureCache::LoadDisplacementTexture(uint32 image)
{
    if (shapeToDisplacement.empty()) LoadDisplacementsForKnownShapes();

    image &= 0x7FFFF;

    sint16 dpWidth;
    sint16 dpHeight;
    std::vector<uint8> displacement;

    // shape is known?
    std::bitset<64 * 64> shape = GetShape(image);
    auto shapeFile = shapeToDisplacement.find(shape);
    if (shapeFile != shapeToDisplacement.end())
    {
        // load from file
        // TODO: move
        char buffer[512];
        platform_get_openrct_data_path(buffer, 512);
        Path::Append(buffer, 512, "displacements");
        Path::Append(buffer, 512, shapeFile->second.c_str());
        String::Append(buffer, 512, ".raw");

        // TODO: error check + assert sizes
        std::ifstream stream(buffer, std::ios::in | std::ios::binary);
        std::vector<uint8> contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        stream.close();

        dpWidth = contents[0];
        dpHeight = contents[1];
        displacement = std::vector<uint8>(dpWidth * dpHeight * 3);
        memcpy(displacement.data(), contents.data() + 2, dpWidth * dpHeight * 3);
    }
    else
    {
        // estimate displacement
        // TODO: move
        rct_drawpixelinfo * dpi = GetImageAsDPI(image, 0);

        dpWidth = dpi->width;
        dpHeight = dpi->height;
        displacement = std::vector<uint8>(dpWidth * dpHeight * 3);

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
        int leftX = dpi->width - 1;

        // find the pixels on the left/right top/bottom "edges" of the bounding box
        //    /T\
        //  LT   RT
        //  /     \
        // |\     /|
        // | \   / |
        //  \ \ / /
        //  LB v RB
        //    \B/
        for (int y = dpi->height - 1; y >= 0; y--) {
            for (int x = 0; x < dpi->width; x++) {
                if (dpi->bits[y * dpi->width + x]) {
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
        int wallHeight = leftWallYBottom - leftWallYTop;
        int wallHeight2 = wallHeight * 2;

        for (int y = dpi->height - 1; y >= 0; y--) {
            for (int x = 0; x < dpi->width; x++) {
                int dx = bottomX - x;
                int dy = ((dpi->height - 1) - y) * 2 - abs(dx);
                off_t offset = dpi->width * 3 * y + 3 * x;
                displacement[offset + 0] = Math::Clamp(0, -dx * 2, 255);
                displacement[offset + 1] = Math::Clamp(0, dx * 2, 255);
                displacement[offset + 2] = Math::Clamp(0, dy / 2, Math::Min(255, wallHeight));
                if (dy > wallHeight2) {
                    displacement[offset + 0] = Math::Clamp(0, (int)displacement[offset + 0] + (dy - wallHeight2), 255);
                    displacement[offset + 1] = Math::Clamp(0, (int)displacement[offset + 1] + (dy - wallHeight2), 255);
                }
            }
        }

        DeleteDPI(dpi);
    }


    auto cacheInfo = AllocateImage(dpWidth, dpHeight);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, cacheInfo.bounds.x, cacheInfo.bounds.y, cacheInfo.index, dpWidth, dpHeight, 1, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, displacement.data());

    return cacheInfo;
}

std::bitset<64 * 64> DisplacementTextureCache::GetShape(uint32 image)
{
    rct_drawpixelinfo * dpi = GetImageAsDPI(image, 0);

    std::bitset<64 * 64> shapeMap;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            if (y < dpi->height && x < dpi->width)
                shapeMap[y * 64 + x] = dpi->bits[y * dpi->width + x] ? 1 : 0;
            else
                shapeMap[y * 64 + x] = 0;
        }

    DeleteDPI(dpi);

    return shapeMap;
}

void DisplacementTextureCache::EnlargeAtlasesTexture(GLuint newEntries)
{
    CreateAtlasesTexture();

    GLuint newIndices = _atlasesTextureIndices + newEntries;

    // Retrieve current array data
    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    auto oldPixels = std::vector<char>(_atlasesTextureDimensions * _atlasesTextureDimensions * _atlasesTextureIndices * 3);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, oldPixels.data());

    // Delete old texture, allocate a new one, then define the new format on the newly created texture
    glDeleteTextures(1, &_atlasesTexture);
    AllocateAtlasesTexture();
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8UI, _atlasesTextureDimensions, _atlasesTextureDimensions, newIndices, 0, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, nullptr);

    // Restore old data
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, _atlasesTextureDimensions, _atlasesTextureDimensions, _atlasesTextureIndices, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, oldPixels.data());

    _atlasesTextureIndices = newIndices;
}




#endif /* DISABLE_OPENGL */
