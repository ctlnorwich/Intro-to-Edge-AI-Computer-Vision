# Intro to Edge AI: Computer Vision
A whirlwind tour of computer vision on embedded Linux. We start with the algorithms that let computers see for the first time, and finish by running a cutting-edge edge AI stack on a Raspberry Pi 5.

      ____     .--.               _____
     | !? |   |o_o |             < OOM >
      `--'    |:_/ |              -----
        \    //   \ \                 \   ^__^
            (|     | )                 \  (oo)\_______
           /'\_   _/` \                   (__)\       )\/\
           \___)=(___/                        ||----w |
                                              ||     ||

This workshop follows on from [Intro to the Unix Terminal](https://github.com/ctlnorwich/Intro-to-the-Unix-Terminal). You should already be happy moving around the terminal, using `nano` or `vim`, setting environment variables and using `uv`, so we won't cover those again here.

**What you need:**

| Thing | Notes |
| --- | --- |
| Raspberry Pi 5 (8GB) | With its fan fitted. Running AI flat out gets the Pi hot! |
| 32GB microSD card | Plus a card reader for your computer |
| 27W USB-C power supply | Weaker supplies can make the Pi limit power to its USB ports |
| USB webcam | Any standard one will do |
| A computer | With [Raspberry Pi Imager](https://www.raspberrypi.com/software/) installed and a web browser |

## Contents
- [Intro to Edge AI: Computer Vision](#intro-to-edge-ai-computer-vision)
  - [Contents](#contents)
  - [What is Edge AI?](#what-is-edge-ai)
  - [Part 1: Flashing the SD Card](#part-1-flashing-the-sd-card)
  - [Ethical Considerations](#ethical-considerations)
    - [A drone in the forest](#a-drone-in-the-forest)
    - [Surveillance](#surveillance)
    - [Questions to keep asking](#questions-to-keep-asking)
  - [Part 2: Booting Up and Getting In](#part-2-booting-up-and-getting-in)
    - [Embedded Linux development: getting into a headless device](#embedded-linux-development-getting-into-a-headless-device)
    - [Connecting with Raspberry Pi Connect](#connecting-with-raspberry-pi-connect)
  - [Part 3: Setting Up the Project](#part-3-setting-up-the-project)
    - [A brief history of computer vision](#a-brief-history-of-computer-vision)
  - [Part 4: Seeing Edges with Sobel (1968)](#part-4-seeing-edges-with-sobel-1968)
    - [Step 1: A still image](#step-1-a-still-image)
    - [Step 2: Live video](#step-2-live-video)
    - [Step 3: Real-time Sobel](#step-3-real-time-sobel)
  - [Part 5: Finding Faces with Viola-Jones (2001)](#part-5-finding-faces-with-viola-jones-2001)
  - [Part 6: Motion Detection with Frame Differencing](#part-6-motion-detection-with-frame-differencing)
  - [Part 7: Edge AI with YOLO26](#part-7-edge-ai-with-yolo26)
    - [Step 1: Export the model to NCNN](#step-1-export-the-model-to-ncnn)
    - [Step 2: Inspect the code](#step-2-inspect-the-code)
    - [Step 3: Run it](#step-3-run-it)
  - [Part 8: CV in the real world](#part-8-cv-in-the-real-world)
    - [Step 1 (everyone): The frame budget](#step-1-everyone-the-frame-budget)
    - [Step 2: Choose your path](#step-2-choose-your-path)
    - [Path A: Follow-me tracker](#path-a-follow-me-tracker)
    - [Path B: Motion-gated detection](#path-b-motion-gated-detection)
    - [Path C: Open-vocabulary detection](#path-c-open-vocabulary-detection)
    - [Path D: Pose estimation and gestures](#path-d-pose-estimation-and-gestures)
  - [Where Next?](#where-next)
  - [Glossary](#glossary)

> [!WARNING]
> **Jargon ahead!** Computer vision and AI come with a lot of jargon. Don't worry if a word goes over your head. Check the [Glossary](#glossary) at the bottom, ask the internet or a technician.

## What is Edge AI?

A Raspberry Pi is a **Single Board Computer** (SBC): a whole computer on a board the size of a credit card. The Pi 5 has four Arm Cortex-A76 cores running at 2.4GHz and 8GB of RAM. That's about the same as a mid-range phone, and roughly what you'd find inside a drone, a robot vacuum or a car.

Today we'll use it as an [embedded system](https://en.wikipedia.org/wiki/Embedded_system): a computer built into something else to do one job. Most of them run Linux on an Arm chip. Developing for them is a bit different from developing for a PC:

- **They're usually headless.** There's no screen or keyboard, so you develop on your computer (the **host**) and connect to the device (the **target**) over a network.
- **Every resource is limited.** Engineers talk about **SWaP-C**: Size, Weight, Power and Cost. A drone can't carry a gaming GPU, so every millisecond and milliwatt counts.

Traditionally AI runs on huge servers in data centres: you send your data off and wait for an answer. **Edge AI** runs the AI on the device itself, on the "edge" of the network. That has some big advantages:

| | Cloud AI | Edge AI |
| --- | --- | --- |
| **Latency** | A round trip to a server, often 100ms or more | Milliseconds |
| **Privacy** | Your video goes to someone else's computer | Your video doesn't need to leave the device |
| **Bandwidth** | Has to stream video constantly | Only sends results, if anything |
| **No signal?** | Stops working | Carries on |

As the Purdue drone team we'll see in a moment puts it: *"closing the loop from pixels to motor actuation in milliseconds right on the edge is what keeps the drone alive and the runner in frame."*

(Today we'll be streaming the Pi's screen to our computers so we can see what it's doing. A finished product wouldn't need to.)

> [!NOTE]
> **Edge AI is not the same as edge detection!** "Edge AI" is about *where* the computing happens (on an edge device like our Pi). "Edge detection", which we do in Part 4, is about finding the *outlines* of objects in a picture.

Our journey today follows the history of computer vision. First we'll use **classical** techniques, where humans refine a technique using algorithms. Then we'll move on to **deep learning**, where the computer learns the maths itself from millions of examples. Finally we'll try and squeeze a modern AI stack onto our tiny Pi.

## Part 1: Flashing the SD Card

The Pi doesn't have a hard drive. Its operating system lives on the microSD card, so first we need to put one on there. We'll use **Raspberry Pi OS**, a version of Debian Linux made for the Pi.

1. Go to [connect.raspberrypi.com](https://connect.raspberrypi.com/) and sign in, or create a free Raspberry Pi ID if you don't have one. We'll use this to reach the Pi later.
2. Put the microSD card in your computer and open **Raspberry Pi Imager**.
3. **Device:** choose **Raspberry Pi 5**.
4. **Operating system:** choose **Raspberry Pi OS (64-bit)**. Make sure it's the version *with* the desktop, because we need to see video windows later.
5. **Storage:** choose your SD card. Double check it's the right one, because everything on it will be wiped!
6. **Customisation:** fill in each page:

   | Page | What to put |
   | --- | --- |
   | Hostname | Something unique like `yourname-cv-system`. There will be lots of Pis in the room! |
   | Localisation | Your capital city, time zone and keyboard layout |
   | User | A username and password. **Write them down!** |
   | Wi-Fi | The network details your technician gives you, or leave it empty if you're using an Ethernet cable |
   | Remote access | Turn on **SSH** with password authentication |
   | Raspberry Pi Connect | Turn it on, click **Open Raspberry Pi Connect**, sign in and create an auth key. Imager fills in the token for you |

7. Click **Write** and wait.

> [!NOTE]
> Imager can't set up "enterprise" networks, which need a username as well as a password and may block the VNC connection. That's why we use a lab network or an Ethernet cable today.

> [!TIP]
> Writing and checking the card takes 5 to 10 minutes. Use that time to read the next section.

## Ethical Considerations

Before we build anything, it's worth thinking about what we're building. Computer vision is one of the most [dual-use](https://en.wikipedia.org/wiki/Dual-use_technology) technologies there is: the same code can be used to help or hurt people.

### A drone in the forest

In October 2026, Professor Aniket Bera from Purdue University's IDEAS Lab [posted a video on LinkedIn](https://www.linkedin.com/posts/abera_purdue-embodiedai-edgeai-ugcPost-7519549199974219777-9AgH/) of a drone chasing a person through a forest ([TechSpot has a decent write-up](https://www.techspot.com/news/114361-watch-autonomous-drone-chase-person-through-forest-without.html)). The drone:

- runs **everything on board**, with no cloud computing
- has **no GPS** (the trees block it) and **no map** of the forest in advance, so it builds its own as it flies
- uses a camera that can see depth (an RGB-D camera) and LiDAR to spot the runner and dodge branches
- **predicts where the person will be** when they disappear behind a tree, so it doesn't lose them

It's an impressive piece of engineering, and it is exactly the kind of edge AI we're learning about today. Now picture two different videos of the very same drone:

1. A mountain biker bombing down a forest trail, filmed by a drone that follows them perfectly. It's a dream shot for an extreme sports video, with no camera operator needed.
2. A soldier running for their life through a forest, followed by a drone that won't lose them.

The drone can't tell the difference, and nothing in the code we write today could either. A few facts to think about:

- **Militaries want this.** Navigating without GPS is exactly what militaries invest in, because GPS can be jammed. In 2025 Purdue announced that the US Army Research Laboratory had signed a five-year, $1.5 million deal with Bera's lab to build drones and ground robots that can find their way through hostile terrain without satellites.
- **Follow-me drones have been here before.** Skydio became famous for consumer drones that could follow and film you on your bike. In [August 2023 it stopped selling them](https://www.skydio.com/blog/skydio-to-sunset-consumer-drone-offerings) to focus on enterprise, public safety and defence customers.
- **Being "on the edge" cuts both ways.** Running the AI on the device protects privacy and works without a signal. That same property means a weapon no longer depends on a radio link that can be jammed, or on a human at the other end of it. In the war in Ukraine, jamming has pushed drones towards finishing their attacks on their own using on-board computer vision. Read more about [lethal autonomous weapons](https://en.wikipedia.org/wiki/Lethal_autonomous_weapon).

### Surveillance

Vision systems are also used to watch people who never agreed to it, and they don't work equally well for everyone:

- **Face detection** (finding faces, which we do in Part 5) is the first step of every **[facial recognition](https://en.wikipedia.org/wiki/Facial_recognition_system)** system (working out *who* the face belongs to). Police in the UK use live facial recognition cameras in public places.
- In the 2018 *Gender Shades* study, [Joy Buolamwini](https://en.wikipedia.org/wiki/Joy_Buolamwini) and Timnit Gebru found that commercial face analysis systems got gender wrong for up to 34.7% of darker-skinned women, but at most 0.8% of lighter-skinned men. AI learns the biases in the data it's trained on.
- In the UK, if your camera films anything outside your own property, data protection law applies to you. See the [ICO's guidance on domestic CCTV](https://ico.org.uk/for-the-public/domestic-cctv-systems/).

### Questions to keep asking

Keep these in mind as you build today. You'll see **CAUTION** boxes coming back to them.

- **Who is in the frame, and did they agree to be?**
- **What happens when it's wrong?** Who pays for a false positive?
- **Who chooses the target?** A person tapping a screen, or the algorithm?
- **If you share your code, who else might use it, and for what?**
- **How would you feel seeing your project on the front page of a newspaper?**

As computer scientists and roboticists you'll be held to codes of conduct like the [ACM Code of Ethics](https://www.acm.org/code-of-ethics). In 2011 UK research councils published the [EPSRC/AHRC Principles of Robotics](https://en.wikipedia.org/wiki/Laws_of_robotics#EPSRC_/_AHRC_principles_of_robotics). The first one reads: *"Robots are multi-use tools. Robots should not be designed solely or primarily to kill or harm humans, except in the interests of national security."* What do you make of that last part...?

> [!IMPORTANT]
> **Ground rules for today:** only point your camera at people who are happy to be filmed. None of our scripts save any video, so please don't change them to.

## Part 2: Booting Up and Getting In

1. Put the microSD card in the Pi.
2. Plug in your USB webcam (and the Ethernet cable, if you're using one).
3. Plug in the power last. The Pi turns on by itself.

The first boot takes a couple of minutes, because the Pi expands its file system to fill the card and then restarts.

### Embedded Linux development: getting into a headless device

Our Pi has no screen, keyboard or mouse, which is how most embedded devices live. There are a few ways in:

| Method | What you get | Good for |
| --- | --- | --- |
| **Serial console** | A terminal over a cable plugged into the board's pins | When the network is broken, or debugging boot problems |
| **SSH** | A terminal over the network | Running commands, scripting, copying files |
| **Remote desktop (VNC)** | The whole desktop, streamed over the network | Anything graphical, like video! |
| **Cloud relay** | Either of the above, through a service on the internet | Reaching a device on a different network, without knowing its IP address |

Computer vision is all about pictures, so we want to *see* what our Pi sees. That means we need a remote desktop. [Raspberry Pi Connect](https://www.raspberrypi.com/documentation/services/connect.html) is a cloud relay that gives us one in the browser. Its screen sharing works just like VNC, but it's built on WebRTC (the same technology as video calls). It connects straight to the Pi where it can, or relays through Raspberry Pi's servers if it can't. That means it usually works even through uni firewalls, and you never need to find the Pi's IP address.

### Connecting with Raspberry Pi Connect

1. Go to [connect.raspberrypi.com](https://connect.raspberrypi.com/). Your Pi should appear with the hostname you gave it.
2. Click **Connect via** and choose **Screen sharing**.
3. A new tab opens with your Pi's desktop. Open the **Terminal** app from the menu.

> [!TIP]
> **What about SSH?** We switched it on in Imager, so you can still use it from your computer's terminal:
> ```bash
> ssh username@yourname-cv-system.local     # .local finds the Pi by its hostname on the local network
> ```
> SSH is perfect for running commands, scripts and copying files (with `scp`). It also lets editors like VS Code work on the Pi remotely. But SSH only gives you text, so the video windows we open today would have nowhere to appear. That's why we use Connect.

> [!TIP]
> Connect also has a **Remote shell** option, which is a terminal in your browser. It even works on Raspberry Pi OS Lite, which has no desktop.

Let's update the list of packages and install a tool for finding cameras:

```bash
sudo apt update
sudo apt install -y v4l-utils
```

> [!TIP]
> On a Pi you're keeping, run `sudo apt full-upgrade` regularly to get security updates. A device on a network that never gets updated is a device waiting to be hacked.

## Part 3: Setting Up the Project

Clone this workshop onto the Pi:

```bash
mkdir -p ~/dev
cd ~/dev
git clone https://github.com/ctlnorwich/Intro-to-Edge-AI-Computer-Vision.git
cd Intro-to-Edge-AI-Computer-Vision
ls
```

The code for each part of the workshop is in its own folder, from `Part-4` to `Part-8`.

> [!IMPORTANT]
> Stay in this top folder (the one with `pyproject.toml` in it) for the rest of the workshop, and put the folder name in front of each script, like `uv run Part-4/sobel_edge.py`. The images and models you download are saved in whichever folder you're in, so this keeps them all in one place where every script can find them. Press <kbd>Tab</kbd> to finish off the folder and file names for you.

Remember `uv` from the Unix workshop? This is a fresh Pi, so it won't be installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

> [!CAUTION]
> Piping something from the internet into `sh` runs whatever it says on your computer. Only ever do this with websites you trust.

This project already has a `pyproject.toml` and a `uv.lock`, so you don't need `uv init` or `uv add`. Have a look at the dependencies:

```bash
cat pyproject.toml
```

| Dependency | What it's for |
| --- | --- |
| `opencv-python` | OpenCV, the classic computer vision library (since 2000) |
| `numpy` | Images are just big grids of numbers, and NumPy does maths on them fast |
| `ultralytics` | The library that runs the YOLO family of AI models |
| `torch`, `torchvision` | PyTorch, the deep learning framework underneath Ultralytics |
| `ncnn`, `pnnx` | Convert models to, and run them in, NCNN: a format built for Arm CPUs |
| `lap` | Maths used by the ByteTrack tracker in Path A |
| `clip` | Turns words into numbers for YOLOE in Path C |

> [!NOTE]
> Look at the `[tool.uv.sources]` part. On Linux, PyTorch normally comes with several GB of libraries for NVIDIA graphics cards. The Pi doesn't have one, so we tell uv to get the CPU-only version instead. That shrinks the download from around 3.4GB to under 400MB. That kind of care is a big part of embedded development: storage, bandwidth and memory are all precious.

Now let uv build the environment exactly as described in `uv.lock`:

```bash
uv sync
```

> [!TIP]
> This downloads a few hundred MB and takes a few minutes. Read about the history of computer vision below while it runs.

### A brief history of computer vision

| Year | Milestone |
| --- | --- |
| 1960 | Rudolf Kálmán publishes the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter), a way to estimate where something is from noisy measurements. It guided the Apollo spacecraft to the Moon, and we'll use it to track people in Part 8: Path A |
| 1966 | MIT's *Summer Vision Project* sets some students the job of making a computer see, over one summer. It [turned out to be a bit harder than that](https://people.csail.mit.edu/brooks/idocs/AIM-100.pdf) |
| 1968 | The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) finds edges using a 'image kernel' a tiny 3x3 grid of numbers. **Part 4** |
| 2001 | [Viola-Jones](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) detects faces in real time, and is soon inside digital cameras. **Part 5** |
| 2005 | [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG) becomes the standard way to detect people |
| 2009 | [ImageNet](https://en.wikipedia.org/wiki/ImageNet), a dataset of millions of labelled photos, is released |
| 2012 | [AlexNet](https://en.wikipedia.org/wiki/AlexNet), a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) trained on gaming GPUs it crushes the ImageNet Large Scale Visual Recognition Challenge. The deep learning era begins |
| 2015 | [YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once) ("You Only Look Once") detects objects in a single pass, fast enough for live video |
| 2021 | [CLIP](https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training) learns to connect pictures with words from 400 million image-caption pairs |
| 2026 | [YOLO26](https://docs.ultralytics.com/models/yolo26/) is released, designed from the start for edge devices, along with YOLOE-26 which detects anything you can describe in words. **Parts 7 and 8** |

## Part 4: Seeing Edges with Sobel (1968)

Before AI, computer vision meant hand-designing maths that turns pixels into something useful. One of the first and most important things to find is **edges**, the outlines of objects. Everything from Viola-Jones in Part 5 to the first layer of YOLO in Part 7 is built on top of them.

### Step 1: A still image

Download a test image of a sudoku puzzle with `curl`. It's full of straight lines, which is perfect for us:

```bash
curl -o test_image.png https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png
```

Open `Part-4/sobel_edge.py` in `nano` or `vim` (or [view it on GitHub](Part-4/sobel_edge.py)) and have a read:

- **`cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)`** throws away the colour. Edges are about light against dark, so colour would just triple the work. (OpenCV stores colours as **B**lue **G**reen **R**ed, rather than the usual RGB, for historical reasons.)
- **`cv2.Sobel(...)`** is the core maths. It slides a 3x3 grid of numbers called a **kernel** over every pixel and calculates the *rate of change* of brightness (the derivative). `sobel_x` finds changes from left to right, which are *vertical* edges. `sobel_y` finds changes from top to bottom, which are *horizontal* edges.
- **`cv2.addWeighted(...)`** blends the two together into one image.

The X kernel looks like this. It subtracts the pixels on the left from the pixels on the right, so it gives a big number wherever the brightness jumps:

```
-1  0  +1
-2  0  +2
-1  0  +1
```

Run it:

```bash
uv run Part-4/sobel_edge.py
```

Four windows open. Compare **Sobel X** and **Sobel Y**: one has only the vertical grid lines and the other only the horizontal ones! Press any key in one of the windows to close them.

> [!NOTE]
> If you see a warning like `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"` but the windows still appear, you can ignore it. OpenCV falls back to another way of drawing windows that works fine. If no windows appear at all, try running it with `QT_QPA_PLATFORM=xcb uv run Part-4/sobel_edge.py`.

### Step 2: Live video

Video is just a fast series of still images, so next let's get pictures from the webcam. First we need to find out which number Linux has given it:

```bash
v4l2-ctl --list-devices
```

The Pi 5 has a few video devices of its own built in, so look for your webcam's *name* and the first `/dev/video` number under it, usually `/dev/video0`. That number is your camera ID.

All of today's scripts read the camera ID from an **environment variable** called `CAMERA`, and use `0` if it isn't set. If your camera isn't `0`, set it for the rest of this terminal session:

```bash
export CAMERA=1
```

Open `Part-4/usb_camera_test.py` in `nano` or `vim` (or [view it on GitHub](Part-4/usb_camera_test.py)) and have a look:

- **`cv2.VideoCapture(cam_id)`** takes control of the camera.
- **`video_capture.set(...)`** asks the camera for a 640x480 picture. Fewer pixels means less work for every step that comes after.
- **`while True:`** loops forever, as fast as the Pi can go.
- **`video_capture.read()`** grabs the latest frame from the camera. A frame is a NumPy array of shape `(480, 640, 3)`: height, width and the three colour channels.
- **`cv2.waitKey(1)`** gives the window 1 millisecond to update, and checks whether you've pressed <kbd>q</kbd>.

```bash
uv run Part-4/usb_camera_test.py
```

Click on the video window and press <kbd>q</kbd> to close it.

### Step 3: Real-time Sobel

Now let's combine the two. Open `Part-4/live_sobel.py` in `nano` or `vim` (or [view it on GitHub](Part-4/live_sobel.py)). Every real-time vision program is built around the same loop:

1. **Read:** grab a frame from the camera.
2. **Process:** do something to it. Here it's the same Sobel maths as Step 1.
3. **Display:** show the result.

```bash
uv run Part-4/live_sobel.py
```

Wave your hand about and watch the outlines move. Press <kbd>q</kbd> to quit.

## Part 5: Finding Faces with Viola-Jones (2001)

Edges tell us *where* things change, but not *what* anything is. In 2001 Paul Viola and Michael Jones published a way to find faces that was fast enough for live video on a 700MHz Pentium III, a computer much slower than our Pi. It worked so well, and was so cheap to run, that it ended up inside the compact digital cameras of the 2000s to focus on people's faces. It was one of the first real-time computer vision systems in a consumer product, and a true embedded system.

It uses four clever ideas:

1. **[Haar-like features](https://en.wikipedia.org/wiki/Haar-like_feature).** Simple rectangles of light and dark. For example, the band across your eyes is usually darker than your cheeks, and the bridge of your nose is lighter than your eyes.
2. **The integral image.** A trick for adding up all the pixels inside any rectangle with just 4 lookups, however big it is.
3. **AdaBoost.** There are over 160,000 possible rectangle features in a small 24x24 window. Training picks out the few hundred that actually matter.
4. **The cascade.** Most of a picture isn't a face. The first stages of the cascade use just a couple of features to throw away obvious non-faces quickly, and only the face-like windows get tested further.

We don't have to train it ourselves, because OpenCV comes with ready-trained cascades. Open `Part-5/face_detect.py` in `nano` or `vim` (or [view it on GitHub](Part-5/face_detect.py)):

- **`cv2.CascadeClassifier(...)`** loads the trained cascade from an XML file.
- **`cv2.equalizeHist(...)`** spreads out the brightness values so faces in dim light still have contrast.
- **`face_cascade.detectMultiScale(...)`** slides a window across the image at lots of different sizes:
  - `scaleFactor=1.1` shrinks the image by 10% each pass, so it can find bigger faces with the same size window.
  - `minNeighbors=5` means a face has to be found by 5 overlapping windows to count, which cuts out false alarms.
  - `minSize=(40, 40)` ignores anything smaller than 40 pixels.

```bash
uv run Part-5/face_detect.py
```

The number in the corner is how long the detection took in milliseconds. **Write it down**, because we'll compare it with a neural network in Part 7.

**Try it yourself:**

- Turn your head to the side, tilt it, or cover one eye. When does it lose you? It was trained on faces looking straight at the camera, with features designed by humans, and it can't cope with much else.
- Change `cascade_file` to `"haarcascade_eye.xml"`, `"haarcascade_smile.xml"` or even `"haarcascade_frontalcatface.xml"`. Run `ls .venv/lib/python3.*/site-packages/cv2/data/` to see them all.
- Change `minNeighbors` to `1`, and then `20`. What happens to the false alarms, and to the real faces?

> [!CAUTION]
> Test it on everyone in your group, with consent. Does it work equally well in different lighting, with glasses, with different skin tones, or with a head covering? A detector is only as fair as the data it was trained on, and the faces used to train this cascade were collected over 20 years ago. Remember this is only face *detection*. Recognising *who* someone is takes just one more step.

> [!NOTE]
> **HOG + SVM (2005).** Four years later, Navneet Dalal and Bill Triggs found a better way to detect people. A [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) splits the image into small cells and counts which *direction* the edges in each cell point (the gradients come from Sobel-style maths!). A [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) then decides whether a window full of those histograms looks like a person or not. It was the standard way to detect pedestrians until deep learning took over. Viola-Jones and HOG are now so old that OpenCV 5 moved them out of the main library into its "contrib" extras. That's why our `pyproject.toml` asks for OpenCV 4.

## Part 6: Motion Detection with Frame Differencing

Another classic trick: if the camera doesn't move, anything that changes between frames must be moving. We take a picture of the empty scene as a **baseline**, then subtract each new frame from it. Any pixels that are different are [foreground](https://en.wikipedia.org/wiki/Foreground_detection), and everything else is background.

Open `Part-6/motion_detect.py` in `nano` or `vim` (or [view it on GitHub](Part-6/motion_detect.py)):

- **The warm-up loop** reads and throws away frames for 2 seconds while the webcam's auto-exposure and white balance settle. The camera only starts streaming on the first `read()`, so just waiting wouldn't help. Grab the baseline too early, and the picture brightening up would look like the whole room moving.
- **`cv2.GaussianBlur(...)`** smooths out camera noise so it isn't mistaken for movement.
- **`cv2.absdiff(baseline, current_frame)`** finds how different each pixel is from the baseline.
- **`cv2.threshold(...)`** turns every pixel that changed by more than 25 (out of 255) pure white and everything else pure black. This is called a **binary mask**.
- **`cv2.dilate(...)`** grows the white areas to fill in any gaps.
- **`cv2.findContours(...)`** finds the outline of each white blob, which we turn into a green box with `cv2.boundingRect(...)`.

> [!IMPORTANT]
> Point the camera at a still background and **step out of the frame** before you run it.

```bash
uv run Part-6/motion_detect.py
```

Walk back into the frame. You'll get two windows: the camera with green boxes, and the black and white mask that shows the maths at work.

**Try it yourself:** turn the lights off and on, or move the camera slightly. What happens?

This is incredibly cheap to run, but it's quite dumb. It tells us *where* something moved, but not *what* it was: a person, a cat or a curtain blowing in the wind all look the same. To know *what* we're looking at, we need AI.

> [!TIP]
> Frame differencing comes back in [Path B](#path-b-motion-gated-detection) of Part 8, where it decides when to wake the AI up.

## Part 7: Edge AI with YOLO26

Viola-Jones used features designed by humans. Deep learning lets a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN) learn its own features from millions of labelled examples. Its first layers end up learning to find edges, just like Sobel. The layers after them combine those into textures and shapes, and the deepest layers into whole objects.

**YOLO** (You Only Look Once) was a breakthrough for real-time AI. Older detectors first guessed where objects *might* be, then checked each guess separately, which was accurate but very slow. YOLO looks at the whole image *once* and predicts every box and its label in a single pass.

We're using **YOLO26n**, the newest and smallest ("n" for nano) Ultralytics model. YOLO26 was designed for edge devices from the start. For example, it can skip **non-maximum suppression** (NMS), an extra clean-up step older YOLOs always needed to remove duplicate boxes. (Our NCNN version still uses it, because NCNN doesn't support the NMS-free mode yet.) It was trained on the [COCO dataset](https://cocodataset.org/), so it knows 80 kinds of object, from `person` and `bicycle` to `cup` and `toothbrush`.

### Step 1: Export the model to NCNN

AI models are usually trained and run with **PyTorch**, which is built for big NVIDIA GPUs. It works on the Pi's CPU, but slowly. **[NCNN](https://github.com/Tencent/ncnn)** is a framework from Tencent made for running neural networks on phones. It's tiny and uses the special maths instructions in Arm CPUs (called NEON). In [Ultralytics' own Raspberry Pi 5 tests](https://www.ultralytics.com/blog/raspberry-pi-object-detection), YOLO26n took 299ms per image with PyTorch and just 67ms with NCNN.

Converting a model to another format is called **exporting** it:

```bash
uv run Part-7/export_ncnn.py
```

This downloads the YOLO26n PyTorch weights (`yolo26n.pt`, about 5MB), then traces the network layer by layer and rewrites it in NCNN's format. Have a look at what it made:

```bash
ls -lh yolo26n_640_ncnn_model
head yolo26n_640_ncnn_model/model.ncnn.param
```

| File | What's in it |
| --- | --- |
| `model.ncnn.param` | A text description of every layer in the network |
| `model.ncnn.bin` | The weights: millions of numbers learned during training |
| `metadata.yaml` | The class names, the image size and other details |

Find the first `Convolution` line in `model.ncnn.param`, just after the `Input`. The very first layer of the network has 16 filters (`0=16`), each 3x3 in size (`1=3`), the same size as the Sobel kernel from 1968. The difference is that this network learned its own numbers.

> [!NOTE]
> An exported NCNN model has a **fixed input size**. The model in `yolo26n_640_ncnn_model` always looks at a 640x640 image. That's why [export_ncnn.py](Part-7/export_ncnn.py) puts the size in the folder name.

> [!NOTE]
> Ultralytics is licensed under the AGPL-3.0, which says that if you build something with it and share it (even as an online service), you have to share your source code too. Licences matter as soon as your project becomes a product.

### Step 2: Inspect the code

Open `Part-7/yolo_detect.py` in `nano` or `vim` (or [view it on GitHub](Part-7/yolo_detect.py)). The Ultralytics library hides nearly all of the neural network maths:

- **`YOLO(model_folder)`** loads the NCNN folder we just made. If the folder isn't there, the script tells you the command to export it.
- **`video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)`** keeps only the newest frame waiting. Normally the camera queues up a few frames, so when the AI is slower than the camera you'd be looking at the past. The delay between something happening and the program reacting is called **latency**, and it isn't the same thing as FPS.
- **`ai_results = detection_model(video_frame)`** passes the frame through the network. You get back one result per image.
- **`ai_results[0].plot()`** draws all the boxes, labels and confidence scores for us.
- **`ai_results[0].speed["inference"]`** is how many milliseconds the neural network itself took.
- **FPS** (frames per second) is measured over the *whole* loop: camera, AI and drawing.

### Step 3: Run it

```bash
uv run Part-7/yolo_detect.py
```

Hold up your phone, a cup/bottle, or a banana. Every box shows a label and a **confidence score** from 0 to 1: how sure the network is.

Look at the numbers in the corner. How do they compare with Viola-Jones? YOLO is finding 80 kinds of object at once, in any pose, but it's probably still a bit laggy. Making it smooth is where Part 8 begins.

> [!NOTE]
> To the network, `person` is just class number 0 of 80, the same as `cup` or `toothbrush`. It has no idea what a person is or why one might matter more than a cup. Any care has to come from us.

## Part 8: CV in the real world

Time to put everything together into the kinds of systems you'll find in the real world. First, everyone measures how fast our model can really go on the Pi. Then you choose a path to build.

### Step 1 (everyone): The frame budget

A webcam gives you 30 frames a second, so you have **33 milliseconds** per frame to do *everything*. On an edge device you'll always be trading accuracy against speed, so let's measure it. Export two smaller versions of the model:

```bash
uv run Part-7/export_ncnn.py yolo26n 320
uv run Part-7/export_ncnn.py yolo26n 160
```

Halving the image size means a quarter of the pixels, and roughly a quarter of the work. Run each one, setting the size with the `IMGSZ` environment variable:

```bash
IMGSZ=640 uv run Part-7/yolo_detect.py
IMGSZ=320 uv run Part-7/yolo_detect.py
IMGSZ=160 uv run Part-7/yolo_detect.py
```

Fill in a table like this for each size, but don't spend more than 10 minutes on it. For the last column, walk backwards away from the camera until it stops seeing you:

| Image size | FPS | AI time (ms) | Confidence on you | How far away it still sees you |
| --- | --- | --- | --- | --- |
| 640 | | | | |
| 320 | | | | |
| 160 | | | | |

Which would you choose for a drone? Or for a camera that counts people across a car park? All of the Part 8 scripts use **320** unless you tell them otherwise, but you can still change it with `IMGSZ`.

> [!TIP]
> Running AI flat out heats the CPU, and when it gets too hot the Pi slows itself down to cool off. This is called **thermal throttling**. Check on it in a second terminal:
> ```bash
> vcgencmd measure_temp     # the CPU temperature
> vcgencmd get_throttled    # throttled=0x0 means it has never had to slow down
> ```

### Step 2: Choose your path

Pick one of these. If you finish, try another!

| Path | You'll build | Used in the real world for | New idea |
| --- | --- | --- | --- |
| [**A: Follow-me tracker**](#path-a-follow-me-tracker) | A tracker that locks onto one person and steers to keep them in shot | Sports filming drones, camera gimbals... and the Purdue drone | Tracking and the Kalman filter |
| [**B: Motion-gated detection**](#path-b-motion-gated-detection) | A system that only wakes the AI up when something moves | Trail cameras, smart doorbells, anything on a battery | Mixing classical CV with AI to save power |
| [**C: Open-vocabulary detection**](#path-c-open-vocabulary-detection) | A detector that finds anything you can describe in words | Searching video, robots that take spoken instructions | Vision-language models |
| [**D: Pose estimation and gestures**](#path-d-pose-estimation-and-gestures) | Gesture controls using your body's skeleton | Sports coaching, fitness apps, gesture-controlled drones | Keypoints |

### Path A: Follow-me tracker

> [!CAUTION]
> This is the most ethically loaded path. What you're about to build is the core of a follow-me sports drone, and also of the drone in the forest. Read the [Ethical Considerations](#ethical-considerations) if you haven't and only track people who have agreed to it.

Detection treats every frame on its own. It can tell you there are two people in this frame, but not which one was which in the last frame. **Tracking** links the detections up over time, giving each object an **ID** that sticks to it.

We'll use **[ByteTrack](https://github.com/FoundationVision/ByteTrack)**. For every object it's following, ByteTrack runs a **[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)**, the same 1960s maths that guided Apollo. The Kalman filter keeps track of each box's position *and* velocity, predicts where the box should be in the next frame, and then matches that prediction to the nearest new detection. Its other trick (the "Byte" in the name) is that it doesn't throw away low-confidence detections. It uses them to hang onto objects that are blurry or half hidden.

When someone disappears completely, ByteTrack keeps their track going for 30 frames as "lost", while the Kalman filter keeps predicting where they've gone. If they reappear near the prediction, they get their old ID back. It's a simple version of the "occlusion prediction" the Purdue drone uses to find people behind trees.

Open `Part-8/follow_me.py` in `nano` or `vim` (or [view it on GitHub](Part-8/follow_me.py)):

- **`detection_model.track(..., persist=True, tracker="bytetrack.yaml")`** runs YOLO26 and then ByteTrack. `persist=True` tells it this frame follows on from the last one.
- **`on_mouse_click`** and **`find_clicked_track`** let you click on a person to choose your target, just like tapping someone in a drone app.
- **`find_predicted_box`** looks inside ByteTrack's list of lost tracks for your target, and gets the Kalman filter's guess of where they've gone.
- **`steering_commands`** works out how far the target is from the centre of the frame (from -1 to +1) and turns it into commands like `YAW LEFT 0.42`. If it's inside the **dead zone** we don't bother moving, otherwise the drone would twitch constantly. A real flight controller would multiply each of these errors by a **gain** to set a motor speed: this is the "P" in a [PID controller](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller), which you'll meet everywhere in robotics.

```bash
uv run Part-8/follow_me.py
```

Every person gets a box and an ID. **Click on someone** to lock on. The arrow shows which way the drone would need to turn, and the commands at the bottom are what it would send to its motors. Press <kbd>c</kbd> to clear the lock and <kbd>q</kbd> to quit.

**Try it yourself:**

- Walk behind a door or a big chair, then come out the other side. While you're hidden, the box turns orange and keeps moving, because the Kalman filter is predicting where you went. Do you get the same ID when you come back out?
- Change `classes_to_track` to `[32]` to follow a sports ball, or `[16]` for a dog.
- Change `dead_zone` and `desired_height`. What would they feel like on a real drone?

> [!CAUTION]
> **Who chooses the target?** Right now a human clicks on someone. Try changing the code so it locks on automatically to the biggest person in view, then discuss it with your neighbour. What has changed? Who is responsible if it picks the wrong person? The idea that a human should always make the final decision about a target is called "meaningful human control", and it's at the centre of the [debate about autonomous weapons](https://en.wikipedia.org/wiki/Lethal_autonomous_weapon).

> [!WARNING]
> Don't connect this to anything that moves towards people, such as motors, drones or robots. Getting that safe is a whole field of engineering of its own.

### Path B: Motion-gated detection

Running a neural network on every frame uses a lot of power and makes a lot of heat. That's fine when the Pi is plugged in, but not for a trail camera that has to last weeks on batteries. The trick is a **cascade**, the same idea as in Viola-Jones: a cheap test runs all the time, and only wakes the expensive one up when it's needed.

Open `Part-8/motion_gate.py` in `nano` or `vim` (or [view it on GitHub](Part-8/motion_gate.py)):

- **Stage 1** is the frame differencing from Part 6, running on every frame.
- With one upgrade: instead of a fixed baseline, **`cv2.accumulateWeighted(...)`** keeps a *running average* of the background, which slowly blends in every new frame. Now lights changing or a chair being moved don't confuse it forever. `background_learning_rate` sets how quickly it forgets.
- **Stage 2** is YOLO26, which only runs while the AI is "awake". It wakes up whenever a big enough blob moves, and stays awake for `stay_awake_seconds` after the last movement.
- The **duty cycle** is the percentage of frames the neural network actually ran on. It's a rough guide to how much work you've saved. (Only rough, because a frame with AI takes longer than one without!)

> [!IMPORTANT]
> Point the camera at a still background and step out of the frame before you run it.

```bash
uv run Part-8/motion_gate.py
```

Watch the duty cycle drop while nothing's happening, and walk in to wake it up.

**Try it yourself:**

- Run `IMGSZ=320 uv run Part-7/yolo_detect.py` for 3 minutes and check `vcgencmd measure_temp`. Let the Pi cool down, then do the same with `motion_gate.py` (which also uses 320) in an empty room. What's the difference?
- Stand completely still. How long until the running average absorbs you into the background and the AI goes back to sleep? Change `background_learning_rate` and try again.
- Swap the running average for OpenCV's more advanced background subtractor, `cv2.createBackgroundSubtractorMOG2()`.

> [!CAUTION]
> Smart doorbells and trail cameras don't just film your front door or a badger's path. They film the street, your neighbours, and people walking past who have no idea. If your camera captures anything outside your own property, UK data protection law applies to you. See the [ICO's guidance](https://ico.org.uk/for-the-public/domestic-cctv-systems/).

### Path C: Open-vocabulary detection

YOLO26 can only find the 80 kinds of object in COCO. If you want it to find something else, you have to collect and label thousands of new pictures and train it again. **Open-vocabulary** models get around this: you just type what you're looking for.

We'll use **[YOLOE-26](https://docs.ultralytics.com/models/yoloe/)** ("Real-Time Seeing Anything"). It's built on YOLO26, plus a text encoder from the [CLIP](https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training) family, which was trained on huge numbers of pictures with captions:

1. The **text encoder** turns each prompt, like `"mobile phone"`, into a list of numbers called an **embedding**. It's like coordinates for its meaning.
2. The vision network makes an embedding for each region of the image, in the same "meaning space".
3. If a region's embedding is close to a prompt's embedding, that's a match.

Once you know which prompts you want, you can **bake** them into the network: the prompt embeddings get folded into the network's weights, and the text encoder isn't needed any more. You can then export it to NCNN like any other YOLO. It's much faster, but it only knows the prompts you baked in.

Open `Part-8/open_vocab.py` in `nano` or `vim` (or [view it on GitHub](Part-8/open_vocab.py)):

- **`YOLOE("yoloe-26n-seg.pt")`** loads the PyTorch model. It does *segmentation* too, outlining the exact pixels of each object, not just a box.
- **`prompt_model.set_classes(prompts)`** runs your words through the text encoder.
- **`bake_to_ncnn(...)`** exports the model with your current prompts built in.

> [!WARNING]
> The first time it runs, YOLOE downloads a 254MB text encoder (`mobileclip2_b.ts`) into the project folder. If the whole room does this at once, the Wi-Fi will have a bad day, so be patient.

Put your prompts after the script name, with quotes around any that contain spaces:

```bash
uv run Part-8/open_vocab.py cup "mobile phone" glasses
```

- Press <kbd>p</kbd> to change the prompts. The video freezes while you type, so **click on the terminal window**, type your prompts separated by commas and press <kbd>Enter</kbd>. Then click back on the video window before pressing any more keys.
- Press <kbd>b</kbd> to bake your current prompts into an NCNN model and switch to it. How much faster is it? Press <kbd>p</kbd> to go back to live prompts.

**Try it yourself:**

- Things that aren't in COCO: `glasses`, `pen`, `hoodie`, `lanyard`, `headphones`, `plant`.
- Descriptions instead of things: `person wearing glasses`, `red mug`, `hand`. Where does it start to break?
- Gibberish, or something that isn't in the room. Does it still find something?

> [!CAUTION]
> Try to imagine someone typing in `suspicious person`. The model might find nothing, but it might well find *something*: whatever looks closest to those words in what it learned from the internet, biases and all. And prompts like `person in a red jacket` move us from detecting *a category* to singling out *an individual*. Being able to describe a target in words is a very powerful thing to put on a camera.

### Path D: Pose estimation and gestures

[Pose estimation](https://en.wikipedia.org/wiki/Articulated_body_pose_estimation) finds the **keypoints** of a body: 17 points for each person, including the nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles. Joined up, they make a skeleton. Coaches use it to analyse sports technique, fitness apps use it to count your reps, and some drones use it so you can control them by waving.

| Keypoint | Number | Keypoint | Number |
| --- | --- | --- | --- |
| Nose | 0 | Left / right wrist | 9 / 10 |
| Left / right eye | 1 / 2 | Left / right hip | 11 / 12 |
| Left / right ear | 3 / 4 | Left / right knee | 13 / 14 |
| Left / right shoulder | 5 / 6 | Left / right ankle | 15 / 16 |
| Left / right elbow | 7 / 8 | | |

First, export the pose version of YOLO26n. It's the same network with a different final layer (called the **head**) that predicts keypoints as well as boxes:

```bash
uv run Part-7/export_ncnn.py yolo26n-pose 320
```

Open `Part-8/pose_gestures.py` in `nano` or `vim` (or [view it on GitHub](Part-8/pose_gestures.py)):

- **`YOLO(model_folder)`** loads the pose model you just exported.
- **`ai_results[0].keypoints.xy`** holds the `[x, y]` position of each keypoint for each person, and **`.conf`** how sure the network is about each one.
- **`hand_raised(...)`** checks whether a wrist is higher than its shoulder. Images count **y downwards** from the top, so "higher" means a *smaller* y!
- **`read_gesture(...)`** turns that into a command, which gets printed in the terminal whenever it changes, ready for a robot to act on.

```bash
uv run Part-8/pose_gestures.py
```

Raise your left hand, your right hand, then both. "Left" means *your* left, which appears on the right of the picture because the camera is facing you.

**Try it yourself:**

- Turn it into a **star jump counter**: count each time both hands go from down to up.
- Add a gesture using your elbows, or both hands on your head (compare the wrists with the nose, keypoint `0`).
- How does it cope when you're side-on to the camera, or someone walks in front of you?

> [!CAUTION]
> Your pose can identify you. The way you walk is so personal that [gait recognition](https://en.wikipedia.org/wiki/Gait_analysis#Gait_as_biometrics) can pick people out even when their faces are covered. Pose tracking is also used to monitor how fast people work in warehouses and offices. Would you want your posture tracked at work?

## Where Next?

You've gone from 1968 maths to a modern edge AI stack, on a computer the size of a credit card. Here's where you could go next:

- **Proper embedding with a camera module.** A [Raspberry Pi Camera Module](https://www.raspberrypi.com/documentation/accessories/camera.html) plugs straight into the Pi's camera connector, and is how you'd build a Pi into a product. You read it with the `picamera2` library instead of OpenCV's `VideoCapture`. The CTL has some to borrow.
- **Seeing in 3D.** The Purdue drone uses a depth camera to know how far away things are. The CTL has [Intel RealSense](https://www.realsenseai.com/) depth cameras. Their colour stream shows up as a normal webcam, so try `CAMERA=` with its number from `v4l2-ctl --list-devices`. To get the depth too, you'll need Intel's `librealsense` library.
- **AI accelerators.** Chips made just for neural networks. The [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) runs models on the camera sensor itself, the [AI HAT+](https://www.raspberrypi.com/products/ai-hat/) adds a Hailo accelerator to the Pi 5, and the [AI HAT+ 2](https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/) is powerful enough for small language and vision-language models.
- **NVIDIA Jetson Orin Nano.** The CTL also has [Jetson Orin Nano](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) developer kits to use. Like the Pi, they're embedded Linux boards with an Arm CPU, but they also have an NVIDIA graphics processor (GPU) built in, giving up to 67 TOPS (trillion operations per second) of AI performance. Boards like this are inside a lot of real robots and drones. Instead of NCNN, you export models to NVIDIA's TensorRT format to run them on the GPU. Ultralytics has a [Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) to get you started. Follow its setup rather than this project's `uv sync`, because our `pyproject.toml` deliberately installs the CPU-only PyTorch on Linux.
- **Squeezing out more speed on the CPU.** Arm has an [optimised runtime for YOLO26n on the Pi 5](https://github.com/arm/Optimized-Runtimes), which Arm measured at around 37ms per image at 640 pixels, against 67ms for NCNN.
- **Training your own model.** If what you need isn't in COCO, you can label your own images and train YOLO26 on them. See the [Ultralytics docs](https://docs.ultralytics.com/modes/train/).
- **Robotics.** [ROS 2](https://docs.ros.org/) is the standard framework for joining up cameras, sensors and motors on real robots, and it runs on the Pi.

And some resources to dig deeper:

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** by Richard Szeliski: the classic textbook, free to read online.
- **[OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)**: everything from filters to feature matching.
- **[Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)**: benchmarks and tips for every export format.
- **[Gender Shades](http://gendershades.org/)**: the study on bias in face analysis, with a great short video.

## Glossary

| Term | Meaning |
| --- | --- |
| **Accelerator** | A chip designed just to run neural networks fast and efficiently, like the Hailo chip on the AI HAT+ |
| **Baseline** | The picture of the empty scene that motion detection compares each new frame against |
| **Binary mask** | An image where every pixel is either pure black or pure white |
| **Bounding box** | The rectangle drawn around a detected object |
| **ByteTrack** | A tracking algorithm that uses a Kalman filter to match boxes between frames, including low-confidence ones |
| **Cascade** | A chain of tests where cheap ones run first and only pass the hard cases on to expensive ones |
| **Class** | A kind of object a model can detect, like `person` or `cup` |
| **Classical computer vision** | Computer vision using maths designed by people, rather than learned from data |
| **CLIP** | A model from 2021 that learned to connect pictures and words from 400 million image-caption pairs |
| **CNN** | Convolutional Neural Network. A neural network that slides small learned kernels over an image |
| **COCO** | Common Objects in Context. A dataset of photos labelled with 80 kinds of everyday object, used to train YOLO26 |
| **Confidence** | How sure a model is about a prediction, from 0 to 1 |
| **Contour** | The outline of a shape in an image |
| **Dead zone** | A range around the target where a controller deliberately does nothing, so it doesn't twitch |
| **Deep learning** | Machine learning using neural networks with many layers |
| **Dual-use** | A technology that can be used for both good and harmful purposes, such as civilian and military |
| **Duty cycle** | The fraction of time something is switched on |
| **Edge AI** | Running AI on the device itself, rather than on a server in the cloud |
| **Edge detection** | Finding the outlines of objects in an image. Nothing to do with Edge AI! |
| **Embedded system** | A computer built into another device to do one job |
| **Embedding** | A list of numbers representing the meaning of a word or a part of an image |
| **Export** | Converting a model into a different format, such as from PyTorch to NCNN |
| **FPS** | Frames per second |
| **Frame** | A single image from a video |
| **Frame budget** | The time you have to process each frame. 33ms at 30 FPS |
| **Frame differencing** | Finding movement by subtracting one frame from another |
| **Gain** | The number a controller multiplies an error by to decide how hard to correct it |
| **Grayscale** | An image with only brightness, no colour |
| **GPU** | Graphics Processing Unit. A chip with thousands of small cores, built for graphics but also great at the maths inside neural networks |
| **Haar-like feature** | A pattern of light and dark rectangles, used by Viola-Jones |
| **Head** | The final layers of a neural network, which turn what it has seen into an answer, like boxes or keypoints |
| **Headless** | A computer with no screen, keyboard or mouse |
| **HOG** | Histogram of Oriented Gradients. A way of describing shapes by which direction their edges point |
| **Host** | The computer you develop on, like your PC |
| **Image size** | The width and height of the square image a model looks at, set with `IMGSZ` today |
| **Inference** | Running a trained model to get predictions, as opposed to training it |
| **Kalman filter** | Maths for estimating where something is and how fast it's moving from noisy measurements, and predicting where it'll be next |
| **Kernel** | A small grid of numbers slid over an image, like the 3x3 Sobel kernel. (Not the same as the Linux kernel!) |
| **Keypoint** | A specific point on an object, like a wrist or a knee |
| **Latency** | The delay between something happening and the system reacting |
| **Model** | A trained neural network: its layout plus its weights |
| **NCNN** | A neural network framework from Tencent, built to run fast on phone and Arm CPUs |
| **NMS** | Non-Maximum Suppression. A clean-up step that removes duplicate boxes around the same object. YOLO26 can run without it |
| **Open vocabulary** | A model that can detect things described in words, rather than a fixed list of classes |
| **Pose estimation** | Finding the positions of a body's keypoints |
| **Prompt** | The text you give an open-vocabulary model to say what to look for |
| **PyTorch** | A popular deep learning framework, used to train most modern models |
| **Raspberry Pi Connect** | Raspberry Pi's service for reaching your Pi's desktop or terminal from a web browser |
| **SBC** | Single Board Computer. A whole computer on one circuit board, like the Raspberry Pi |
| **Segmentation** | Finding the exact pixels that belong to each object, rather than just a box |
| **SSH** | Secure Shell. An encrypted way to use another computer's terminal over a network |
| **SVM** | Support Vector Machine. A classical machine learning method that finds the best line (or surface) separating two groups |
| **SWaP-C** | Size, Weight, Power and Cost. The limits every embedded system is designed around |
| **Target** | The device you are developing for, like our Pi. Also, the thing a tracker follows |
| **TensorRT** | NVIDIA's format and runtime for running neural networks fast on its GPUs, like the one in a Jetson |
| **Thermal throttling** | When a CPU slows itself down to stop overheating |
| **Threshold** | A cut-off value. Above it is one thing, below it is another |
| **TOPS** | Trillions of Operations Per Second. A rough measure of how fast an AI chip is |
| **Tracking** | Following objects from frame to frame, giving each one an ID |
| **Viola-Jones** | The 2001 method for detecting faces in real time using Haar-like features and a cascade |
| **VNC** | Virtual Network Computing. A way to see and control another computer's desktop over a network |
| **Wayland** | The modern system Raspberry Pi OS uses to draw windows on the screen |
| **WebRTC** | The technology behind video calls in your browser, which Raspberry Pi Connect uses for screen sharing |
| **Weights** | The millions of numbers inside a neural network that are learned during training |
| **YOLO** | You Only Look Once. A family of fast object detection models that look at the whole image in a single pass |
