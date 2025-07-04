## Computational design and fabrication of reusable multi-tangent bar structures.
![image](https://github.com/KIKI007/Scaffold/blob/main/rhino/teaser.png)

## Installations

### 1. Obtain the free academic license for Gurobi.

1. [Register for a free Gurobi account as an academic and log in](https://portal.gurobi.com/iam/register/).

2. [Following the installation instructions](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)

### 2. Install Scaffold Python Package

You may install the software in a python or conda virtual environment.
The minimum requirement for python is version 3.9.

```bash
pip install -e .
python -m pip install 'pytransform3d[all]'
```

To run optimization:
```bash
python .\script\optimize.py --name one_tet
```

To visualize optimized results:
```bash
python .\script\visualization.py --name one_tet
```

### 4. Rhino plugin

The recent rhino 8 support native python.
```bash
C:\Users\USERNAME\.rhinocode\py39-rh8\python.exe -m pip install -e .
```
Replace USERNAME to your local folder name. If such a folder does not exists, please run `_ScriptEditor` command in rhino 8 to initialize the rhino-python environment.

To run test example,
1. Open the rhino file `rhino\test_example.3dm`
2. Run the script `rhino\run_script.py` using rhino script editor.
3. Choose whether to use your own model or load from our example folder.
4. If load your own model
   1. Confirm the parameters list
   2. Select input curves
   3. Select boundary curves
   4. Wait until optimization finished
5. If load from example folder
    1. Choose a model to optimize
    2. Wait until optimization finished

![image](https://github.com/KIKI007/Scaffold/blob/main/rhino/example.gif)

## Hardware

We use the wood dowel with outer diameter 20mm.

We use the [F14TAD swivel coupler](https://shop.globaltruss.de/en/TRUSSING/Deco-truss/F14/Swivel-coupler-for-F14.html?listtype=search&searchparam=SWIVEL%20COUPLER) from global truss ([technical drawing](https://shop.globaltruss.de/out/media/F14TAD_TZ_Trussaufnehmer_doppelt.pdf)).