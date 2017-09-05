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

#include <vector>
#include <stdexcept>
#include <openrct2/core/Memory.hpp>
#include "TextureCache.h"
#include "openrct2/core/Math.hpp"

extern "C"
{
    #include <openrct2/drawing/drawing.h>
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

CachedTextureInfo DisplacementTextureCache::LoadDisplacementTexture(uint32 image)
{
    rct_drawpixelinfo * dpi = GetImageAsDPI(image, 0);

    std::vector<uint8> displacement = std::vector<uint8>(dpi->width * dpi->height * 3);
    
    // experimental
    int wallLX = -1, wallLY = -1, wallLH = 0, wallRX = -1, wallRY = -1, wallRH = 0;

    for (int x = 0; x < dpi->width - 1; x++) {
        for (int y = dpi->height - 1; y >= 0; y--)
            if (dpi->bits[y * dpi->width + x]) {
                wallLX = x;
                if (wallLY < 0) wallLY = y;
                wallLH = y - wallLY;
            }
        if (wallLY >= 0) break;
    }

    if (wallLH >= -4) wallLH = 0;

    for (int x = dpi->width - 1; x >= 1; x--) {
        for (int y = dpi->height - 1; y >= 0; y--)
            if (dpi->bits[y * dpi->width + x]) {
                wallRX = x;
                if (wallRY < 0) wallRY = y;
                wallRH = y - wallRY;
            }
        if (wallRY >= 0) break;
    }

    if (wallRH >= -4) wallRH = 0;

    // bottom pixel X value
    int bottomX = dpi->width / 2, bottomY = dpi->height - 1, topY = -1;
    int bottomWH = (wallLH + wallRH) / 2; // todo lerp

    for (int y = dpi->height - 1; y >= 0; y--) {
        for (int x = 0; x < dpi->width; x++) {
            int firstX = -1, lastX = -1;
            if (dpi->bits[y * dpi->width + x]) {
                if (firstX == -1) firstX = x;
                lastX = x;
            }
            if (firstX >= 0) {
                bottomX = (firstX + lastX) / 2;
                bottomY = y;
                goto foundBPY;
            }
        }
    }
foundBPY:

    for (int y = 0; y < dpi->height; y++) {
        if (dpi->bits[y * dpi->width + bottomX]) {
            topY = y;
            break;
        }
    }

    int maxDX = Math::Max(bottomX - wallLX, wallRX - bottomX);
    int extraTopWH = ((bottomY + bottomWH) - topY) - maxDX;
    extraTopWH = (extraTopWH + 8) / 16 * 16;
    int midY = bottomY + bottomWH;

    // walls
    for (int y = 0; y < dpi->height; y++) {
        for (int x = 0; x < dpi->width; x++) {
            int dx = x - bottomX;
            float wh;
            float wht;
            float whb;

            if (dx < 0)
            {
                float frac = Math::Clamp(0.0f, (float)(bottomX - x) / (bottomX - wallLX), 1.0f);
                wh = (wallLY + wallLH) * frac + (1 - frac) * (bottomY + bottomWH);
                wht = (wallLY + wallLH) * frac + (1 - frac) * (bottomY + bottomWH - extraTopWH);
                whb = (wallLY) * frac + (1 - frac) * (bottomY);
                displacement[y * dpi->width * 3 + x * 3 + 0] = 0;
                displacement[y * dpi->width * 3 + x * 3 + 1] = -dx * 2;
            }
            else
            {
                float frac = Math::Clamp(0.0f, (float)(x - bottomX) / (wallRX - bottomX), 1.0f);
                wh = (wallRY + wallRH) * frac + (1 - frac) * (bottomY + bottomWH);
                wht = (wallRY + wallRH) * frac + (1 - frac) * (bottomY + bottomWH - extraTopWH);
                whb = (wallRY) * frac + (1 - frac) * (bottomY);
                displacement[y * dpi->width * 3 + x * 3 + 0] = dx * 2;
                displacement[y * dpi->width * 3 + x * 3 + 1] = 0;
            }

            // is above wall?
            if (y < wh) {
                // above wall
                float fracy = (float)(y - midY) / (topY - midY);
                float targetTop = wh * (1 - fracy) + wht * fracy;

                displacement[y * dpi->width * 3 + x * 3 + 2] = Math::Clamp(0.0f, whb - targetTop, 255.0f);
                float ratio = 1.0f / ((bottomY - topY) / (float)maxDX);
                float dp = wh - y;
                displacement[y * dpi->width * 3 + x * 3 + 0] += dp * 2 * ratio;
                displacement[y * dpi->width * 3 + x * 3 + 1] += dp * 2 * ratio;
            }
            else
            {
                displacement[y * dpi->width * 3 + x * 3 + 2] = Math::Clamp(0.0f, whb - y, 255.0f);
            }
        }
    }


    auto cacheInfo = AllocateImage(dpi->width, dpi->height);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _atlasesTexture);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, cacheInfo.bounds.x, cacheInfo.bounds.y, cacheInfo.index, dpi->width, dpi->height, 1, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, displacement.data());

    DeleteDPI(dpi);

    return cacheInfo;
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
