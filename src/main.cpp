#include "main.h"
#include "preview.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

static bool streamCompaction = true;
static bool sortByMat = false;
static bool cacheFirstBounce = false;
static int antiAliasing = NOAA;
static bool viewEdges = false;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

void printState(bool printStream = true, bool printMat = true, bool printCache = true, bool printAA = true, bool printEdge = true) {
	if (printStream && printMat && printCache && printAA && printEdge)
		printf("----- Settings -----\n");

	if (printStream)
		printf("Stream compaction: %s\n", streamCompaction ? "On" : "Off");

	if (printMat)
		printf("Sort by material: %s\n", sortByMat ? "True" : "False");

	if (printCache)
		printf("Cache first bounce: %s\n", cacheFirstBounce ? "True" : "False");
	
	if (printAA)
		printf("Anti Aliasing: %s\n", antiAliasing == NOAA ? "None" : (antiAliasing == AA ? "Stochastic Supersampling" : "Adaptive Supersampling"));

	if (printEdge) {
		if (antiAliasing == ADAPTIVE || (printStream && printMat && printCache && printAA && printEdge)) {
			printf("Edge View: %s\n", viewEdges ? "On" : "Off");
		}
		else {
			printf("Can only toggle edges in adaptive sampling mode!\n");
		}
	}

	printf("\n--------------------\n");
	printf("\n");
}

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char *sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera &cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	//Print starting settings
	printState();

	// Initialize CUDA and GL components
	init();

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, pix);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera &cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene, antiAliasing);
	}

	if (iteration < renderState->iterations) {
		uchar4 *pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration, sortByMat, cacheFirstBounce, viewEdges, streamCompaction);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
		
		if (iteration == 1000) {
			saveImage();
		}
	}
	else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_T:
			streamCompaction = !streamCompaction;
			printState(true, false, false, false, false);
			break;
		case GLFW_KEY_A:
			iteration = 0;
			antiAliasing = (antiAliasing + 1) % 3;
			if (antiAliasing == NOAA) {
				viewEdges = false;
			}
			printState(false, false, false, true, false);
			break;
		case GLFW_KEY_M:
			sortByMat = !sortByMat;
			printState(false, true, false, false, false);
			break;
		case GLFW_KEY_C:
			cacheFirstBounce = !cacheFirstBounce;
			printState(false, false, true, false, false);
			break;
		case GLFW_KEY_E:
			if (antiAliasing == ADAPTIVE) {
				viewEdges = !viewEdges;
			}
			printState(false, false, false, false, true);
			break;
		case GLFW_KEY_SPACE:
			iteration = 0;
			streamCompaction = true;
			sortByMat = false;
			cacheFirstBounce = false;
			antiAliasing = NOAA;
			viewEdges = false;
			printState();
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera &cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
