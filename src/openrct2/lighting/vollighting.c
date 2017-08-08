#include "vollighting.h"
#include "../world/map.h"
#include "../world/scenery.h"
#include "../world/footpath.h"

lighting_value gMapLighting[64][512][512];

// how the light will be affected when light passes through a certain plane
lighting_value lightingAffectorsX[lightmap_size_y][lightmap_size_x][lightmap_size_z];
lighting_value lightingAffectorsY[lightmap_size_y][lightmap_size_x][lightmap_size_z];
lighting_value lightingAffectorsZ[lightmap_size_y][lightmap_size_x][lightmap_size_z];

// lightmap columns whose lightingAffectors are outdated
// each element contains a bitmap, each bit identifying if a direction should be recomputed
// Z is always recomputed
uint8 affectorRecomputeQueue[lightmap_size_y][lightmap_size_x];

lighting_chunk lightingChunks[lightmap_chunks_z][lightmap_chunks_y][lightmap_chunks_x];

const lighting_value black = { .r = 0,.g = 0,.b = 0 };
const lighting_value ambient = { .r = 20,.g = 20,.b = 20 };
const lighting_value lit = { .r = 255,.g = 255,.b = 255 };

bool didInit2 = false;

// multiplies @target light with some multiplier light value @apply
void lighting_multiply(lighting_value* target, const lighting_value apply) {
	target->r *= apply.r / 255.0f;
	target->g *= apply.g / 255.0f;
	target->b *= apply.b / 255.0f;
}

// adds @target light with some multiplier light value @apply
void lighting_add(lighting_value* target, const lighting_value apply) {
	// TODO: can overflow
	target->r += apply.r;
	target->g += apply.g;
	target->b += apply.b;
}

// elementswise lerp between @a and @b depending on @frac (@lerp = 0 -> @a)
// @return 
lighting_value interpolate_lighting(const lighting_value a, const lighting_value b, float frac) {
	return (lighting_value) { 
		.r = a.r * (1.0f - frac) + b.r * frac,
		.g = a.g * (1.0f - frac) + b.g * frac,
		.b = a.b * (1.0f - frac) + b.b * frac
	};
}

// cast a ray from a 3d world position @a (light source position) to lighting tile
// @return 
lighting_value FASTCALL lighting_raycast(lighting_value color, const rct_xyz32 light_source_pos, const rct_xyz16 lightmap_texel) {
	float x = light_source_pos.x / 16.0f;
	float y = light_source_pos.y / 16.0f;
	float z = light_source_pos.z / 2.0f;
	float dx = (lightmap_texel.x + .5) - x;
	float dy = (lightmap_texel.y + .5) - y;
	float dz = (lightmap_texel.z + .5) - z;

	for (int px = min(lightmap_texel.x+1, (int)ceil(x)); px <= max((int)floor(x), lightmap_texel.x); px++) {
		float multiplier = (px - x) / dx;
		float py = dy * multiplier + y - .5;
		float pz = dz * multiplier + z - .5;
		int affectorY = py;
		int affectorZ = pz;
		float progy = py - affectorY;
		float progz = pz - affectorZ;
		lighting_value v00 = lightingAffectorsX[affectorY][px][affectorZ];
		lighting_value v01 = lightingAffectorsX[affectorY+1][px][affectorZ];
		lighting_value v10 = lightingAffectorsX[affectorY][px][affectorZ + 1];
		lighting_value v11 = lightingAffectorsX[affectorY+1][px][affectorZ + 1];
		lighting_value v0 = interpolate_lighting(v00, v01, progy);
		lighting_value v1 = interpolate_lighting(v10, v11, progy);
		lighting_multiply(&color, interpolate_lighting(v0, v1, progz));
	}

	for (int py = min(lightmap_texel.y + 1, (int)ceil(y)); py <= max((int)floor(y), lightmap_texel.y); py++) {
		float multiplier = (py - y) / dy;
		float px = dx * multiplier + x - .5;
		float pz = dz * multiplier + z - .5;
		int affectorX = px;
		int affectorZ = pz;
		float progx = px - affectorX;
		float progz = pz - affectorZ;
		lighting_value v00 = lightingAffectorsY[py][affectorX][affectorZ];
		lighting_value v01 = lightingAffectorsY[py][affectorX + 1][affectorZ];
		lighting_value v10 = lightingAffectorsY[py][affectorX][affectorZ + 1];
		lighting_value v11 = lightingAffectorsY[py][affectorX + 1][affectorZ + 1];
		lighting_value v0 = interpolate_lighting(v00, v01, progx);
		lighting_value v1 = interpolate_lighting(v10, v11, progx);
		lighting_multiply(&color, interpolate_lighting(v0, v1, progz));
	}

	for (int pz = min(lightmap_texel.z + 1, (int)ceil(z)); pz <= max((int)floor(z), lightmap_texel.z); pz++) {
		float multiplier = (pz - z) / dz;
		float px = dx * multiplier + x - .5;
		float py = dy * multiplier + y - .5;
		int affectorX = px;
		int affectorY = py;
		float progx = px - affectorX;
		float progy = py - affectorY;
		lighting_value v00 = lightingAffectorsZ[affectorY][affectorX][pz];
		lighting_value v01 = lightingAffectorsZ[affectorY][affectorX + 1][pz];
		lighting_value v10 = lightingAffectorsZ[affectorY + 1][affectorX][pz];
		lighting_value v11 = lightingAffectorsZ[affectorY + 1][affectorX + 1][pz];
		lighting_value v0 = interpolate_lighting(v00, v01, progx);
		lighting_value v1 = interpolate_lighting(v10, v11, progx);
		lighting_multiply(&color, interpolate_lighting(v0, v1, progy));
	}

	return color;
}

// inserts a static light into the chunks this light can reach
void lighting_insert_static_light(const lighting_light light) {
	int range = 11;
	sint32 lm_x = light.map_x * 2;
	sint32 lm_y = light.map_y * 2;
	sint32 lm_z = light.pos.z / 2;
	for (int sz = max(0, (lm_z - range * 2) / lightmap_chunk_size); sz <= min(lightmap_chunks_z - 1, (lm_z + range * 2) / lightmap_chunk_size); sz++) {
		for (int sy = max(0, (lm_y - range) / lightmap_chunk_size); sy <= min(lightmap_chunks_y - 1, (lm_y + range) / lightmap_chunk_size); sy++) {
			for (int sx = max(0, (lm_x - range) / lightmap_chunk_size); sx <= min(lightmap_chunks_x - 1, (lm_x + range) / lightmap_chunk_size); sx++) {
				lighting_chunk* chunk = &lightingChunks[sz][sy][sx];
				// TODO: bounds check
				chunk->static_lights[chunk->static_lights_count++] = light;
			}
		}
	}
}

void lighting_invalidate_at(sint32 wx, sint32 wy) {
	// remove static lights at this position
	// iterate chunks lights could reach, find lights in this column, remove them
	int range = 11;
	sint32 lm_x = wx * 2;
	sint32 lm_y = wy * 2;
	for (int sz = 0; sz < lightmap_chunks_z; sz++) {
		for (int sy = max(0, (lm_y - range) / lightmap_chunk_size); sy <= min(lightmap_chunks_y - 1, (lm_y + range) / lightmap_chunk_size); sy++) {
			for (int sx = max(0, (lm_x - range) / lightmap_chunk_size); sx <= min(lightmap_chunks_x - 1, (lm_x + range) / lightmap_chunk_size); sx++) {
				lighting_chunk* chunk = &lightingChunks[sz][sy][sx];
				for (size_t lidx = 0; lidx < chunk->static_lights_count; lidx++) {
					if (chunk->static_lights[lidx].map_x == wx && chunk->static_lights[lidx].map_y == wy) {
						chunk->static_lights[lidx] = chunk->static_lights[chunk->static_lights_count - 1];
						chunk->static_lights_count--;
						lidx--;
					}
				}
				chunk->invalid = true;
			}
		}
	}

	// iterate column
	rct_map_element* map_element = map_get_first_element_at(wx, wy);
	if (map_element) {
		do {
			switch (map_element_get_type(map_element)) {
				case MAP_ELEMENT_TYPE_PATH: {
					if (footpath_element_has_path_scenery(map_element) && !(map_element->flags & MAP_ELEMENT_FLAG_BROKEN)) {
						// test
						rct_scenery_entry *sceneryEntry = get_footpath_item_entry(footpath_element_get_path_scenery_index(map_element));
						sint32 x = wx * 32 + 16;
						sint32 y = wy * 32 + 16;
						if (sceneryEntry->path_bit.flags & PATH_BIT_FLAG_LAMP) {
							int z = map_element->base_height * 2;
							if (!(map_element->properties.path.edges & (1 << 0))) {
								lighting_insert_static_light((lighting_light) { .map_x = wx, .map_y = wy, .pos = { .x = x - 14, .y = y, .z = z }, .color = lit });
							}
							if (!(map_element->properties.path.edges & (1 << 1))) {
								lighting_insert_static_light((lighting_light) { .map_x = wx, .map_y = wy, .pos = { .x = x, .y = y + 14, .z = z }, .color = lit });
							}
							if (!(map_element->properties.path.edges & (1 << 2))) {
								lighting_insert_static_light((lighting_light) { .map_x = wx, .map_y = wy, .pos = { .x = x + 14, .y = y, .z = z }, .color = lit });
							}
							if (!(map_element->properties.path.edges & (1 << 3))) {
								lighting_insert_static_light((lighting_light) { .map_x = wx, .map_y = wy, .pos = { .x = x, .y = y - 14, .z = z }, .color = lit });
							}
						}
					}
					break;
				}
			}
		} while (!map_element_is_last_for_tile(map_element++));
	}

	int x = wx * 2;
	int y = wy * 2;

	// revert values to lit...
	for (int z = 0; z < lightmap_size_z; z++) {
		for (int ulm_y = y; ulm_y <= y + 2; ulm_y++) {
			for (int ulm_x = x; ulm_x <= x + 2; ulm_x++) {
				lightingAffectorsX[y][ulm_x][z] = lit;
				lightingAffectorsX[y+1][ulm_x][z] = lit;
				lightingAffectorsY[ulm_y][x][z] = lit;
				lightingAffectorsY[ulm_y][x+1][z] = lit;
			}
		}
		lightingAffectorsZ[y][x][z] = lit;
		lightingAffectorsZ[y + 1][x][z] = lit;
		lightingAffectorsZ[y][x + 1][z] = lit;
		lightingAffectorsZ[y + 1][x + 1][z] = lit;
	}

	// queue rebuilding affectors
	affectorRecomputeQueue[y][x] = 0b1111;
	affectorRecomputeQueue[y+1][x] = 0b1111;
	affectorRecomputeQueue[y][x+1] = 0b1111;
	affectorRecomputeQueue[y+1][x+1] = 0b1111;
}

void lighting_init() {
	// reset affectors to 1^3
	for (int z = 0; z < 64; z++) {
		for (int y = 0; y < 512; y++) {
			for (int x = 0; x < 512; x++) {
				lightingAffectorsX[y][x][z] = lit;
				lightingAffectorsY[y][x][z] = lit;
				lightingAffectorsZ[y][x][z] = lit;
			}
		}
	}

	// init chunks
	for (int z = 0; z < lightmap_chunks_z; z++) {
		for (int y = 0; y < lightmap_chunks_y; y++) {
			for (int x = 0; x < lightmap_chunks_x; x++) {
				lightingChunks[z][y][x].invalid = true;
				lightingChunks[z][y][x].static_lights_count = 0;
				lightingChunks[z][y][x].x = x;
				lightingChunks[z][y][x].y = y;
				lightingChunks[z][y][x].z = z;
			}
		}
	}

	lighting_reset();

	didInit2 = false;
}

void lighting_invalidate_all() {
	// invalidate/recompute all columns ((re)loads all lights on the map)
	for (int y = 0; y < MAXIMUM_MAP_SIZE_PRACTICAL-1; y++) {
		for (int x = 0; x < MAXIMUM_MAP_SIZE_PRACTICAL-1; x++) {
			lighting_invalidate_at(x, y);
		}
	}
}

#pragma region Colormap
// stolen from opengldrawingengine
static const float TransparentColourTable[144 - 44][3] =
{
	{ 0.7f, 0.8f, 0.8f }, // 44
	{ 0.7f, 0.8f, 0.8f },
	{ 0.3f, 0.4f, 0.4f },
	{ 0.2f, 0.3f, 0.3f },
	{ 0.1f, 0.2f, 0.2f },
	{ 0.4f, 0.5f, 0.5f },
	{ 0.3f, 0.4f, 0.4f },
	{ 0.4f, 0.5f, 0.5f },
	{ 0.4f, 0.5f, 0.5f },
	{ 0.3f, 0.4f, 0.4f },
	{ 0.6f, 0.7f, 0.7f },
	{ 0.3f, 0.5f, 0.9f },
	{ 0.1f, 0.3f, 0.8f },
	{ 0.5f, 0.7f, 0.9f },
	{ 0.6f, 0.2f, 0.2f },
	{ 0.5f, 0.1f, 0.1f },
	{ 0.8f, 0.4f, 0.4f },
	{ 0.3f, 0.5f, 0.4f },
	{ 0.2f, 0.4f, 0.2f },
	{ 0.5f, 0.7f, 0.5f },
	{ 0.5f, 0.5f, 0.7f },
	{ 0.3f, 0.3f, 0.5f },
	{ 0.6f, 0.6f, 0.8f },
	{ 0.5f, 0.5f, 0.2f },
	{ 0.4f, 0.4f, 0.1f },
	{ 0.7f, 0.7f, 0.4f },
	{ 0.7f, 0.5f, 0.3f },
	{ 0.6f, 0.4f, 0.2f },
	{ 0.8f, 0.7f, 0.4f },
	{ 0.8f, 0.7f, 0.1f },
	{ 0.7f, 0.4f, 0.0f },
	{ 1.0f, 0.9f, 0.2f },
	{ 0.4f, 0.6f, 0.2f },
	{ 0.3f, 0.4f, 0.2f },
	{ 0.5f, 0.7f, 0.3f },
	{ 0.5f, 0.6f, 0.4f },
	{ 0.4f, 0.4f, 0.3f },
	{ 0.7f, 0.8f, 0.5f },
	{ 0.3f, 0.7f, 0.2f },
	{ 0.2f, 0.6f, 0.0f },
	{ 0.4f, 0.8f, 0.3f },
	{ 0.8f, 0.5f, 0.4f },
	{ 0.7f, 0.4f, 0.3f },
	{ 0.9f, 0.7f, 0.5f },
	{ 0.5f, 0.3f, 0.7f },
	{ 0.4f, 0.2f, 0.6f },
	{ 0.7f, 0.5f, 0.8f },
	{ 0.9f, 0.0f, 0.0f },
	{ 0.7f, 0.0f, 0.0f },
	{ 1.0f, 0.3f, 0.3f },
	{ 1.0f, 0.4f, 0.1f },
	{ 0.9f, 0.3f, 0.0f },
	{ 1.0f, 0.6f, 0.3f },
	{ 0.2f, 0.6f, 0.6f },
	{ 0.0f, 0.4f, 0.4f },
	{ 0.4f, 0.7f, 0.7f },
	{ 0.9f, 0.2f, 0.6f },
	{ 0.6f, 0.1f, 0.4f },
	{ 1.0f, 0.5f, 0.7f },
	{ 0.6f, 0.5f, 0.4f },
	{ 0.4f, 0.3f, 0.2f },
	{ 0.7f, 0.7f, 0.6f },
	{ 0.9f, 0.6f, 0.6f },
	{ 0.8f, 0.5f, 0.5f },
	{ 1.0f, 0.7f, 0.7f },
	{ 0.7f, 0.8f, 0.8f },
	{ 0.5f, 0.6f, 0.6f },
	{ 0.9f, 1.0f, 1.0f },
	{ 0.2f, 0.3f, 0.3f },
	{ 0.4f, 0.5f, 0.5f },
	{ 0.7f, 0.8f, 0.8f },
	{ 0.2f, 0.3f, 0.5f },
	{ 0.5f, 0.5f, 0.7f },
	{ 0.5f, 0.3f, 0.7f },
	{ 0.1f, 0.3f, 0.7f },
	{ 0.3f, 0.5f, 0.9f },
	{ 0.6f, 0.8f, 1.0f },
	{ 0.2f, 0.6f, 0.6f },
	{ 0.5f, 0.8f, 0.8f },
	{ 0.1f, 0.5f, 0.0f },
	{ 0.3f, 0.5f, 0.4f },
	{ 0.4f, 0.6f, 0.2f },
	{ 0.3f, 0.7f, 0.2f },
	{ 0.5f, 0.6f, 0.4f },
	{ 0.5f, 0.5f, 0.2f },
	{ 1.0f, 0.9f, 0.2f },
	{ 0.8f, 0.7f, 0.1f },
	{ 0.6f, 0.3f, 0.0f },
	{ 1.0f, 0.4f, 0.1f },
	{ 0.7f, 0.3f, 0.0f },
	{ 0.7f, 0.5f, 0.3f },
	{ 0.5f, 0.3f, 0.1f },
	{ 0.5f, 0.4f, 0.3f },
	{ 0.8f, 0.5f, 0.4f },
	{ 0.6f, 0.2f, 0.2f },
	{ 0.6f, 0.0f, 0.0f },
	{ 0.9f, 0.0f, 0.0f },
	{ 0.6f, 0.1f, 0.3f },
	{ 0.9f, 0.2f, 0.6f },
	{ 0.9f, 0.6f, 0.6f },
};
#pragma endregion Colormap

void lighting_static_light_cast(lighting_value* target_value, lighting_light light, sint32 px, sint32 py, sint32 pz) {
	sint32 range = 11;
	sint32 w_x = px * 16 + 8;
	sint32 w_y = py * 16 + 8;
	sint32 w_z = pz * 2 + 1;
	float distpot = sqrt((w_x - light.pos.x)*(w_x - light.pos.x) + (w_y - light.pos.y)*(w_y - light.pos.y) + (w_z - light.pos.z)*(w_z - light.pos.z) * 4 * 4);
	float intensity = 1.0f - distpot / (range * 12.0f);
	if (intensity > 0) {
		rct_xyz16 target = { .x = px,.y = py,.z = pz };
		intensity *= 10;
		lighting_value source_value = { .r = intensity,.g = intensity,.b = intensity };
		lighting_add(target_value, lighting_raycast(source_value, light.pos, target));
	}
}

lighting_chunk* lighting_update_affectors() {
	for (int y = 0; y < 512; y++) {
		for (int x = 0; x < 512; x++) {
			uint8 dirs = affectorRecomputeQueue[y][x];
			if (dirs) {
				rct_map_element* map_element = map_get_first_element_at(x / 2, y / 2);
				if (map_element) {
					do {
						switch (map_element_get_type(map_element))
						{
							case MAP_ELEMENT_TYPE_SURFACE: {
								for (int z = 0; z < map_element->base_height-1; z++) {
									lightingAffectorsX[y][x][z] = black;
									lightingAffectorsY[y][x][z] = black;
									lightingAffectorsX[y][x+1][z] = black;
									lightingAffectorsY[y+1][x][z] = black;
									lightingAffectorsZ[y][x][z] = black;
									lightingAffectorsZ[y][x][z + 1] = black;
								}
								break;
							}
							case MAP_ELEMENT_TYPE_SCENERY: {
								for (int z = map_element->base_height - 1; z < map_element->clearance_height - 1; z++) {
									lightingAffectorsX[y][x][z] = black;
									lightingAffectorsY[y][x][z] = black;
									lightingAffectorsX[y][x + 1][z] = black;
									lightingAffectorsY[y + 1][x][z] = black;
									lightingAffectorsZ[y][x][z] = black;
									lightingAffectorsZ[y][x][z + 1] = black;
								}
								break;
							}
							case MAP_ELEMENT_TYPE_WALL: {
								for (int z = map_element->base_height - 1; z < map_element->clearance_height; z++) {
									lighting_value affector = black;
									if (map_element->properties.wall.type == 54) {
										uint8 color = map_element->properties.wall.colour_1;
										if (color >= 44 && color < 144) {
											affector.r = TransparentColourTable[color - 44][0] * 255;
											affector.g = TransparentColourTable[color - 44][1] * 255;
											affector.b = TransparentColourTable[color - 44][2] * 255;
										}
									}
									switch (map_element_get_direction(map_element)) {
									case MAP_ELEMENT_DIRECTION_NORTH:
										if (y % 2 == 1) lightingAffectorsY[y + 1][x][z] = affector;
										break;
									case MAP_ELEMENT_DIRECTION_SOUTH:
										if (y % 2 == 0) lightingAffectorsY[y][x][z] = affector;
										break;
									case MAP_ELEMENT_DIRECTION_EAST:
										if (x % 2 == 1) lightingAffectorsX[y][x + 1][z] = affector;
										break;
									case MAP_ELEMENT_DIRECTION_WEST:
										if (x % 2 == 0) lightingAffectorsX[y][x][z] = affector;
										break;
									}
								}
								break;
							}
						}
					} while (!map_element_is_last_for_tile(map_element++));
				}
				affectorRecomputeQueue[y][x] = 0;
			}
		}
	}

	for (int z = 0; z < lightmap_chunks_z; z++) {
		for (int y = 0; y < lightmap_chunks_y; y++) {
			for (int x = 0; x < lightmap_chunks_x; x++) {
				if (lightingChunks[z][y][x].invalid) {
					// rebuild lightmap
					lighting_chunk* chunk = &lightingChunks[z][y][x];
					for (int oz = 0; oz < lightmap_chunk_size; oz++) {
						for (int oy = 0; oy < lightmap_chunk_size; oy++) {
							for (int ox = 0; ox < lightmap_chunk_size; ox++) {
								chunk->data[oz][oy][ox] = ambient;
								for (size_t lidx = 0; lidx < chunk->static_lights_count; lidx++) {
									//chunk->data[oz][oy][ox] = lit;
									//log_info("got light");
									lighting_static_light_cast(&chunk->data[oz][oy][ox], chunk->static_lights[lidx], x*lightmap_chunk_size + ox, y*lightmap_chunk_size + oy, z*lightmap_chunk_size + oz);
								}
								gMapLighting[z*lightmap_chunk_size + oz][y*lightmap_chunk_size + oy][x*lightmap_chunk_size + ox] = chunk->data[oz][oy][ox];
							}
						}
					}
					chunk->invalid = false;

					// send chunk data to gpu
					return chunk;
				}
			}
		}
	}

	return 0;
}

void lighting_reset() {
	for (int y = 0; y < 512; y++) {
		for (int x = 0; x < 512; x++) {
			if (y < 256 && x < 256)
				affectorRecomputeQueue[y][x] = 0b1111;

			for (int z = 0; z < 64; z++) {
				gMapLighting[z][y][x] = ambient;
			}
		}
	}
}

lighting_chunk* lighting_update() {
	//if (!lighting_update_affectors()) return 0;
	//didInit2 = true;
	//log_info("upd ltn");
	//return (uint8*)&gMapLighting[0][0][0];
	return lighting_update_affectors();
}