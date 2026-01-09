# Initial Domain Generation

To reproduce the generation of initial domain snapshots and uncontrolled statistics,
please follow these steps.

First, we generate the initial domain snapshots using this command, e.g. for ```CylinderJet2D-easy-v0```:
```python
python runscripts/preparation/create_initial_domains.py -m \
    env_id=CylinderJet2D-easy-v0
```

Then, we collect domain statistics using uncontrolled episode rollouts:
```python
python runscripts/preparation/collect_statistics.py -m \
    overwrite_existing=false \
    env_id=CylinderJet2D-easy-v0
```

To aggregate the statistics run:
```python
python runscripts/preparation/aggregate_statistics.py -m \
    env_id=CylinderJet2D-easy-v0
```

For the TCF environments, we additionally collect opposition control episodes using:
```python
python runscripts/preparation/collect_opposition_control.py -m \
    overwrite_existing=false \
    env_id=TCFSmall3D-both-easy-v0
```