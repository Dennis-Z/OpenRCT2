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

#include "DrawImageShader.h"

DrawImageShader::DrawImageShader() : OpenGLShaderProgram("drawimage")
{
    GetLocations();

    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_vboInstances);
    glGenVertexArrays(1, &_vao);

    GLuint vertices[] = { 0, 1, 2, 2, 1, 3 };
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindVertexArray(_vao);

    glVertexAttribIPointer(vIndex, 1, GL_INT, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, _vboInstances);
    glVertexAttribIPointer(vClip, 4, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, clip));
    glVertexAttribIPointer(vTexColourAtlas, 1, GL_INT, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, texColourAtlas));
    glVertexAttribPointer(vTexColourBounds, 4, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, texColourBounds));
    glVertexAttribIPointer(vTexMaskAtlas, 1, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, texMaskAtlas));
    glVertexAttribPointer(vTexMaskBounds, 4, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, texMaskBounds));
    glVertexAttribIPointer(vTexPaletteAtlas, 1, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, texPaletteAtlas));
    glVertexAttribPointer(vTexPaletteBounds, 4, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, texPaletteBounds));
    glVertexAttribIPointer(vFlags, 1, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, flags));
    glVertexAttribPointer(vColour, 4, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, colour));
    glVertexAttribIPointer(vBounds, 4, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, bounds));
    glVertexAttribIPointer(vMask, 1, GL_INT, sizeof(DrawImageInstance), (void*) offsetof(DrawImageInstance, mask));
	glVertexAttribPointer(vPrelight, 1, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, prelight));
	glVertexAttribPointer(vWorldBoxOrigin, 3, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, worldBoxOrigin));
    glVertexAttribIPointer(vTexDisplacementAtlas, 1, GL_INT, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, texDisplacementAtlas));
    glVertexAttribPointer(vTexDisplacementBounds, 4, GL_FLOAT, GL_FALSE, sizeof(DrawImageInstance), (void*)offsetof(DrawImageInstance, texDisplacementBounds));

    glEnableVertexAttribArray(vIndex);
    glEnableVertexAttribArray(vClip);
    glEnableVertexAttribArray(vTexColourAtlas);
    glEnableVertexAttribArray(vTexColourBounds);
    glEnableVertexAttribArray(vTexDisplacementAtlas);
    glEnableVertexAttribArray(vTexDisplacementBounds);
    glEnableVertexAttribArray(vTexMaskAtlas);
    glEnableVertexAttribArray(vTexMaskBounds);
    glEnableVertexAttribArray(vTexPaletteAtlas);
    glEnableVertexAttribArray(vTexPaletteBounds);
    glEnableVertexAttribArray(vFlags);
    glEnableVertexAttribArray(vColour);
    glEnableVertexAttribArray(vBounds);
    glEnableVertexAttribArray(vMask);
	glEnableVertexAttribArray(vPrelight);
	glEnableVertexAttribArray(vWorldBoxOrigin);

    glVertexAttribDivisor(vClip, 1);
    glVertexAttribDivisor(vTexColourAtlas, 1);
    glVertexAttribDivisor(vTexColourBounds, 1);
    glVertexAttribDivisor(vTexDisplacementAtlas, 1);
    glVertexAttribDivisor(vTexDisplacementBounds, 1);
    glVertexAttribDivisor(vTexMaskAtlas, 1);
    glVertexAttribDivisor(vTexMaskBounds, 1);
    glVertexAttribDivisor(vTexPaletteAtlas, 1);
    glVertexAttribDivisor(vTexPaletteBounds, 1);
    glVertexAttribDivisor(vFlags, 1);
    glVertexAttribDivisor(vColour, 1);
    glVertexAttribDivisor(vBounds, 1);
    glVertexAttribDivisor(vMask, 1);
	glVertexAttribDivisor(vPrelight, 1);
	glVertexAttribDivisor(vWorldBoxOrigin, 1);

    Use();
    glUniform1i(uTexture, 0);
    glUniform1i(uDisplacementTexture, 1);
    glUniform1i(uLightmap, 2);
    glUniform1i(uLightmapInterpolate, 3);
}

DrawImageShader::~DrawImageShader()
{
    glDeleteBuffers(1, &_vbo);
    glDeleteBuffers(1, &_vboInstances);
    glDeleteVertexArrays(1, &_vao);
}

void DrawImageShader::GetLocations()
{
    uScreenSize         = GetUniformLocation("uScreenSize");
    uTexture            = GetUniformLocation("uTexture");
    uDisplacementTexture      = GetUniformLocation("uDisplacementTexture");
    uPalette            = GetUniformLocation("uPalette");
    uLightmap           = GetUniformLocation("uLightmap");
    uLightmapInterpolate      = GetUniformLocation("uLightmapInterpolate");
    uRotationTransform  = GetUniformLocation("uRotationTransform");

    vIndex              = GetAttributeLocation("vIndex");
    vClip               = GetAttributeLocation("ivClip");
    vTexColourAtlas     = GetAttributeLocation("ivTexColourAtlas");
    vTexColourBounds    = GetAttributeLocation("ivTexColourBounds");
    vTexDisplacementAtlas = GetAttributeLocation("ivTexDisplacementAtlas"); 
    vTexDisplacementBounds    = GetAttributeLocation("ivTexDisplacementBounds");
    vTexMaskAtlas       = GetAttributeLocation("ivTexMaskAtlas");
    vTexMaskBounds      = GetAttributeLocation("ivTexMaskBounds");
    vTexPaletteAtlas    = GetAttributeLocation("ivTexPaletteAtlas");
    vTexPaletteBounds   = GetAttributeLocation("ivTexPaletteBounds");
    vFlags              = GetAttributeLocation("ivFlags");
    vColour             = GetAttributeLocation("ivColour");
    vBounds             = GetAttributeLocation("ivBounds");
    vMask               = GetAttributeLocation("ivMask");
	vPrelight           = GetAttributeLocation("ivPrelight");
	vWorldBoxOrigin     = GetAttributeLocation("ivWorldBoxOrigin");
}

void DrawImageShader::SetScreenSize(sint32 width, sint32 height)
{
    glUniform2i(uScreenSize, width, height);
}

void DrawImageShader::SetPalette(const vec4f *glPalette)
{
    glUniform4fv(uPalette, 256, (const GLfloat *) glPalette);
}

void DrawImageShader::SetRotationTransform(const float rotationTransform[4])
{
    glUniformMatrix2fv(uRotationTransform, 1, GL_TRUE, rotationTransform);
}

void DrawImageShader::DrawInstances(const std::vector<DrawImageInstance>& instances)
{
    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vboInstances);
    glBufferData(GL_ARRAY_BUFFER, sizeof(instances[0]) * instances.size(), instances.data(), GL_STREAM_DRAW);

    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)instances.size());
}

#endif /* DISABLE_OPENGL */
