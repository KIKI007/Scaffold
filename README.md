--env=GRB_CLIENT_LOG=3# FrameX

## Installations

### 1. Clone and download to your host machine (Windows 🪟)

Let's start by cloning this the repository into a directory named `YOUR_FOLDER/FrameX`, where `YOUR_FOLDER` denotes any folder you want to save this software:
```
git clone --recursive git@github.com:KIKI007/FrameX.git
```
Notice that we have cloned this repository using the `--recursive` flag to make sure we have all the git submodules in place.
If you forgot to include the `--recursive`, issue `git submodule update --init --recursive`.

Now you are done with procedures from the host machine (Windows)! Now, let's switch to Docker, where we will build a minimal Linux virtual machine that allows us to compile and run the code.

### 2. Build the docker image 🐳

Download and install [docker desktop](https://www.docker.com/products/docker-desktop).

In your PowerShell terminal, navigate to the directory where you clone FrameX and build the docker container by:

```bash
cd "YOUR_FOLDER/FrameX"
docker build -t framex .
```

### 3. Obtain the free academic license for Gurobi

[Register for a free Gurobi account as an academic and log in](https://portal.gurobi.com/iam/register/).

<!-- foldable markdown -->
<details>
  <summary>Click to expand for details on downloading a license!</summary>

Download the license file `gurobi.lic` (picture below) and place it in a folder of your choice. You will need to mount this folder to the docker container later.

![](./docs/images/gurobi_license_download.png)

</details>

### 4. Set up X11 server to enable GUI from the docker container

Download and install [vcxsrv](https://sourceforge.net/projects/vcxsrv/).
Launch `XLaunch` and follow the instructions. Make sure you check the `Disable access control` box. You should see a little X icon in your taskbar.
<!-- https://medium.com/@potatowagon/how-to-use-gui-apps-in-linux-docker-container-from-windows-host-485d3e1c64a3 -->

### 5. Launch the docker container 🐳

First, get the IPv4 address of your host machine by commanding `ipconfig`.

Issue the following in the command line, replace `<license parent folder>` and `<your IPv4 address>` with the actual path to the folder where you put the `gurobi.lic` file and your IPv4 address respectively:

```bash
docker run -it --rm -v ${pwd}/../FrameX/:/FrameX -v <license parent folder>/gurobi.lic:/opt/gurobi/gurobi.lic:ro -e DISPLAY=<your IPv4 address>:0.0 framex
```

And you should see something like (the long string of numbers and letters is the container ID and it will be different for you):

```
root@33f7e2f89d8a:/
```

This means you are now inside the docker container, a pure world like the Shire!

First, let's check the gui forwarding is working. Issue `xeyes` and you should see a pair of eyes popping up on your screen.

Then, issue:
```
python3 -c "import pybullet as p; p.connect(p.GUI); input()"
``` 
and you should see a simulation window popping up.

### 6. Install FrameX 🚀

In the docker container, run following commands:
```bash
cd FrameX && pip install -e . -v
```

To test if things are installed correctly, issue
```
python3 scripts/run_smilp_contactopt.py -p one_tet_0622.json --viewer
```

If you see something that end with:
```
Solution status:  0
Collision-free clamp pose found!
data saved to /FrameX/data/multi_tangent_data/mt_results/one_tet_MT.json
```

Congrats 🥳 You are done with the installation!

Note ⚠️: 

The `--rm` flag in the `docker run` line means that if you exit the docker container (with `ctrl+d`), the container will be automatically removed. This means that whenever you relaunch a new container (it's a fresh new environment!), you will need to rerun the line with `pip install -e .` again.

## Usage (v2.0)

In the first phase of the project, we will start with a workflow that roughly takes the following steps:

1. (GH) Draw a 3D line graph in Rhino, pick them up in GH
2. (GH) define bars to be fixed. Fixed bars will not be changed during the optimization.
3. (GH) Change the file name and click a button to export a frame JSON file
4. (docker command line 🐳) In a docker command line prompt, run a python script that reads that JSON file, does the computation, and write result to another JSON file.
5. (GH) Back in GH, refresh so that it picks up the newly computed multi-tangent structure.

Detailed instructions:

Steps 1-3: open the GH file `E:\Code\MAS_T3\FrameX\data\multi_tangent_data\parse_mt_structure_json.gh` in your Grasshopper. Follow the instructions there to set up geometries and parameters.  
![](./docs/images/v1.0_GH_input.png)

Step 4: on docker command line prompt, issue
```
python3 scripts/run_smilp_contactopt.py -p <your json file name> --viewer
```
![](./docs/images/v1.0_GH_docker_command.png)
![](./docs/images/v1.0_GH_docker_finish.png)

Step 5: back in GH, refresh to pick up the computed geometry
![](./docs/images/v1.0_GH_result.png)

Step 6: (optional) on docker command line prompt, issue
```
python3 scripts/run_smilp_adjustopt.py -p <your json file name> --viewer
```
![](./docs/images/v1.0_GH_docker_command.png)
![](./docs/images/v1.0_GH_docker_finish.png)

## Cloud Computation
It is possible to use our server to accelerate computation.

### 1. Setup your server account
Please contact Ziqi for this step, and he will also install the required software.
In windows powershell 
```bash
ssh yihung@tars.inf.ethz.ch
```
### 2. Update your repository

```bash
cd FrameX
git pull origin master
```
### 3. Switch to your data repository
```bash
cd data/multi_tangent_data
git checkout issue-dev
git pull origin issue-dev
```

### 4. Setup docker
```bash
cd ../../
docker build -t framex .
docker run -it --rm -v $PWD/../FrameX/:/FrameX -v ~/gurobi.lic:/opt/gurobi/gurobi.lic:ro framex
```

### 5. Run Program
```bash
cd FrameX && pip install -e .
python3 scripts/optimizer.py -p <your JSON file> --viewer
```

After this initial computation, you can run an adjustment optimization to improve the result.
```bash
python3 scripts/run_smilp_adjustopt.py -p <your JSON file> --viewer
```

### 6. Save results
Step 1. Exist your docker
```bash
exit
```
Step 2. commit change
```bash
git pull origin master
git add * && git commit -m "from server"
git push origin
cd data/multi_tangent_data
git add * && git commit -m "from server"
git push origin issue-dev
```
Congrats 🥳 You have learned cloud computing 🌬️!

### 7. Validate results locally and reading the collision diagnosis

Because we can't effectively forward GUI from server to your local machine, you need to validate the results in your docker locally. 
```bash
python3 scripts/run_validation.py -p <your MT JSON file> --viewer
```

If something is wrong, the pybullet GUI will show you the collision diagnosis like this:

![](./docs/images/vailidate_coupler_bar.png)

This tells you which part is colliding with which part. In this case, because the coupler geometry is not explicitly encoded in the optimization, the coupler's bulk is colliding with a bar.
One way to fix this is to increase the `smallest distance between two bars` to force bars to stay further.

💡 in pybullet's viewer, you can pan the camera by holding `alt` and dragging the mouse. 
`alt + left click` to rotate the camera.
`alt + right click` to zoom in and out.
`alt + middle click` to move the camera up and down.
You can also zoom in and out by scrolling the mouse wheel.

Press `Enter` to proceed to the next step. You might encounter other collisions like this one:

![](./docs/images/vailidate_bar_bar.png)

In this case, two bars that are not connected to the same joint in the original design graph is colliding. 
This is because we only model collisions between bars that are connected to the same joint in the original design graph in the optimization.
And the available bar length is too long so that the one "far-away" bar is sticking into another one.
One way to fix this is to decrease the `available bar length` to force bars to stay shorter (but in reality we won't always be able to do this) or play with the geometry a bit more.

## Troubleshooting

### My docker commandline is frozen!
Close the frozen command prompt and open a new one. Issue `docker ps` and find the name of your docker container in use, which should look sth like `0aa94223e3a1`.
Then issue `docker attach 0aa94223e3a1` and you are back on track!

## Hardware

We use the wood dowel with outer diameter 20mm.

We use the [F14TAD swivel coupler](https://shop.globaltruss.de/en/TRUSSING/Deco-truss/F14/Swivel-coupler-for-F14.html?listtype=search&searchparam=SWIVEL%20COUPLER) from global truss ([technical drawing](https://shop.globaltruss.de/out/media/F14TAD_TZ_Trussaufnehmer_doppelt.pdf)).


[//]: # (## Depedencies)

[//]: # (Before cmake and build, add `export KNITRODIR=<path to your knitro folder>` to your bash profile &#40;`~/.bashrc`&#41;.)

[//]: # ()
[//]: # (You need to install knitro's python package in order for our `multi_tangent` python package to work. Please follow [these instructions]&#40;https://www.artelys.com/docs/knitro/2_userGuide/gettingStarted/startPython.html&#41; to install it.)

[//]: # (All the other python dependencies are handled automatically by pip.)

[//]: # (## Testing)

[//]: # (To test python modules, issue the following:)

[//]: # (```)

[//]: # (pip install -r requirements-dev.txt -v)

[//]: # (pytest)

[//]: # (```)
