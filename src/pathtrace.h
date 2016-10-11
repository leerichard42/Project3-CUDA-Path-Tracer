#pragma once

#include <vector>
#include "scene.h"

#define NOAA 0
#define AA 1
#define ADAPTIVE 2

void pathtraceInit(Scene *scene, int antiAliasing);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool sortByMat, bool cacheFirstBounce, bool viewEdges, bool streamCompaction);
