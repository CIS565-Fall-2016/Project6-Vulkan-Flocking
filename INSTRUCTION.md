Vulkan Demo - Flocking
============================

This is due Tuesday, November 15

**Summary:** In this project, you will be guided through the workings of a
basic Vulkan compute-and-shading application, a 2D version of the Boids
algorithm you implemented back in
[Project 1](https://github.com/CIS565-Fall-2016/Project1-CUDA-Flocking).
Concepts covered will include setting up basic compute and graphics pipelines,
setting up vertex buffers that can be shared between both pipelines,
creating commands for each pipeline, and binding information to each pipeline.

This project is built on top of Sascha Willems's fantastic
[Vulkan examples and demos](https://github.com/SaschaWillems/Vulkan) project,
which contains code examples of many more Vulkan features not covered in this
project.

## Part 0: Setting up for Vulkan
This project requires a GPU compatible with Vulkan 1.0. You will need the
LunarG Vulkan SDK available [here](https://vulkan.lunarg.com/).
For a walkthrough on installing the SDK and verifying/troubleshooting Vulkan
compatibility, take a look at the Vulkan SDK sections of this tutorial
[here](https://vulkan-tutorial.com/Development_environment).

Basically, you're going to want to install the SDK and see if the `cube.exe`
in the SDK directories runs.

## Part 1: Some Vulkan Concepts
In a Vulkan application, the CPU submits commands to queues on the GPU.
Tasks in a GPU queue may include memory copies, allocations, runs through
graphics or compute pipelines, synchronization commands...

We will only explicitly submit runs through a graphics pipeline, runs through
a compute pipeline, and fences to ensure that the graphics pipeline doesn't
start drawing a frame until we're done computing that frame.

### Vulkan Queues
A Vulkan application may have multiple queues for different GPU tasks, and these
queues in turn can be backed by different parts of the same GPU or even hardware
on different GPUs altogether.

Sascha Willems's `VulkanExampleBase` class, which our `computeparticles` boids
app builds on top of, handles selecting devices and creating the graphics queue.
The provided `prepareCompute` function prepares our compute queue.

### Pipelines, Descriptor Layouts, and Descriptor Sets
At a high level, a Vulkan pipeline contains a bunch of state, some shaders,
and some descriptor layouts.

If a pipeline is like a function call from a Python program  to a C function,
a Descriptor Layout is kind of like a description of what order the C function
expects arguments to be provided in.
It basically describes on the CPU side how your shader (analogous to a kernel)
expects information to be provided.

A Descriptor Set is like a set of arguments that you can bind to a pipeline with
a compatible Descriptor Layout. The descriptor sets in our example can be seen
as encapsulating buffers and how the buffers should be interpreted as attributes
in the shaders. Some examples of how Descriptor Sets and Layouts will work in our
application:
* Sharing boid positions and velocities between the graphics pipeline and compute
pipeline involves a descriptor layout for the compute pipeline and a similar
concept for the graphics pipeline, a `vertexInputAttributeDescription` for
each vertex attribute going into the graphics pipeline.
* Ping-Ponging buffers in the compute pipeline involves a descriptor set for each
'flip-flop'

A Descriptor Layout has to be provided to a pipeline when it is created, while
Descriptor Sets can be created somewhat more flexibly.

### Command Buffers and fences
A Vulkan Command Buffer contains a "recording" of a bunch of Vulkan API calls,
such as rendering pass configuration like setting scissor tests and initiation
of passes through graphics and compute pipelines.
A Command Buffer will also describe a binding of a Descriptor Set to an invoked
pipeline.
Since a Command Buffer is a "recording" that gets submitted to a queue on the GPU,
commands can be recorded and re-used. In our example case, the overhead of just
recreating the commands for kicking off drawing and kicking off compute seems to
be relatively low.

Vulkan commands in a queue do not necessarily run sequentially, and commands in
different queues are even less likely to run synchronously. Fences allow us to
ensure that each compute command we run is completed before the GPU starts work
on the next compute command in the queue.
Fences also allow synchronization between a queue and the CPU side application.

### Command Buffer Pools and Descriptor Set Pools
Vulkan uses "pools" of memory for Command Buffers and Descriptor Sets to help
the GPU avoid constantly having to manage memory for these things.
In our example assignment, we have relatively few command buffers and
descriptor sets.

### Rendering
Vulkan also allows very explicit control over how image buffering when
rendering works. Sascha Willems's base code largely covers this.

### Validation Layers
Vulkan calls only return useful errors if validation layers are enabled.
We have enabled these for you by default.

## Part 2: Boids in Vulkan!
At a high level, our code will do the following:
* set up some buffers to represent our boids
* set up a graphics pipeline and describe how it takes in boid data
* set up a compute pipeline and describe how it takes in boid data
* create two sets of "arguments" that we can pass to the compute pipeline

* when running, record Command Buffers that can be placed into the Vulkan Queue

## 2.1: Code Tour

Follow the LOOK labels in the following files:
* `vulkanBoids.cpp`
* `particle.comp`
* `particle.vert`
* `particle.frag`

Don't worry about reading and understanding every single line of code, just try
to understand what each LOOK block describes in terms of how Vulkan controls the
GPU (commands encapsulating pipelines + buffers, running on Queues).

## 2.2: Getting movement

Complete the TODOs `vulkanBoids.cpp`.
When you have randomized velocities, the second descriptorSet, and
flip-flopping done, you should be able to at least see some movement.

Try disabling different blocks of code you wrote for each TODO.
Observe what happens.

## 2.3: Let there be flocking!
Implement the boids algorithm in `data/shaders/computeparticles/particle.comp`.
Keep in mind that whenever you change these shaders, you will need to regenerate
SPIRV versions of these using the `generate-spirv.bat` script.

## Part 3: README.md
Include a GIF of your final simulation. Answer the following questions:

* Why do you think Vulkan expects explicit descriptors for things like
generating pipelines and commands? HINT: this may relate to something in the
comments about some components using pre-allocated GPU memory.
* Describe a situation besides flip-flop buffers in which you may need multiple
descriptor sets to fit one descriptor layout.
* What are some problems to keep in mind when using multiple Vulkan queues?
  * take into consideration that different queues may be backed by different hardware
  * take into consideration that the same buffer may be used across multiple queues
* What is one advantage of using compute commands that can share data with a
rendering pipeline?

## Part 4: Enrichment?
For more details on how the Vulkan rendering pipeline works, we strongly
encourage you to check out the tutorial at vulkan-tutorial.com.

If you want to tackle adding features to your Vulkan flocking, such as mouse
interaction or an extension to 3D flocking, a good place to take a first look
is Sascha Willems's original Vulkan compute particle simulation.
We built his project off that simulation, which demonstrates how to use
uniforms with the rendering pipeline and how to use the framework's mouse and
keyboard interaction.
