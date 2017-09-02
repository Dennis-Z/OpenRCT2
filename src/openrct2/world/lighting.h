#pragma once

#include <time.h>
#include "../common.h"
#include "../world/map.h"

#define LIGHTING_CELL_SUBDIVISIONS 2
#define LIGHTMAP_SIZE_X (256 * LIGHTING_CELL_SUBDIVISIONS)
#define LIGHTMAP_SIZE_Y (256 * LIGHTING_CELL_SUBDIVISIONS)
#define LIGHTMAP_SIZE_Z 128
#define LIGHTMAP_CHUNK_SIZE 16
#define LIGHTMAP_CHUNKS_X (LIGHTMAP_SIZE_X / LIGHTMAP_CHUNK_SIZE)
#define LIGHTMAP_CHUNKS_Y (LIGHTMAP_SIZE_Y / LIGHTMAP_CHUNK_SIZE)
#define LIGHTMAP_CHUNKS_Z (LIGHTMAP_SIZE_Z / LIGHTMAP_CHUNK_SIZE)
#define LIGHTING_MAX_CHUNKS_LIGHTS 64
#define LIGHTING_MAX_CLOCKS_PER_FRAME (CLOCKS_PER_SEC / 100)
#define LIGHTING_MAX_CHUNK_UPDATES_PER_FRAME 100

#pragma pack(push, 1)
typedef struct lighting_value {
	uint8 r;
	uint8 g;
	uint8 b;
} lighting_value;
#pragma pack(pop)

typedef struct lighting_light {
	rct_xyz32 pos;
	lighting_value color;
	sint32 map_x;
	sint32 map_y;
} lighting_light;

typedef struct lighting_chunk {
    lighting_value data[LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE];
    lighting_value data_dynamic[LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE];
    lighting_value skylight_carry[LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE];
    lighting_light static_lights[LIGHTING_MAX_CHUNKS_LIGHTS];
	size_t static_lights_count;
	uint8 x, y, z;
    bool invalid;
    bool has_dynamic_lights;
} lighting_chunk;

typedef struct lighting_update_batch {
	lighting_chunk* updated_chunks[LIGHTING_MAX_CHUNK_UPDATES_PER_FRAME + 1];
    size_t update_count;
} lighting_update_batch;

void lighting_init();
void lighting_reset();
void lighting_invalidate_at(sint32 x, sint32 y);
void lighting_invalidate_around(sint32 x, sint32 y);
void lighting_invalidate_all();
lighting_update_batch lighting_update();