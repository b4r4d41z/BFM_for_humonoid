# ROS bag to canonical Zarr

The converter treats each input bag as one episode. Arm-state message timestamps
are the transition clock. Hand state, recorded arm command, recorded hand command,
and configured cameras are matched by nearest timestamp within `--max-skew-s`.
It never interpolates or resamples; a missing match fails conversion. The last
state supplies `next_obs` for the preceding transition, so an input with `N`
state samples produces `N-1` transitions and only the final transition is done.

Copy `configs/data_contract/rosbag_topics.example.json` and replace every vector
`field` with the confirmed dotted ROS message attribute containing values in the
project's exact order. The repository does not establish those message fields or
the streams' units. Consequently, `scale`, optional `offset`, and `units` are
explicit configuration rather than inferred conversions. The example's camera
roles are deliberately neutral; assign semantic roles only when independently
confirmed.

Camera inputs must be `sensor_msgs/CompressedImage`. They are decoded to RGB and
stored as `uint8 [T,H,W,3]` arrays under `images/<configured-role>`, which favors
training-time random access over preserving transport encoding. All frames for a
camera must have one resolution.

```bash
python scripts/data/convert_bag_to_zarr.py \
  --input /path/to/episode.bag \
  --output /path/to/episode.zarr \
  --topics /path/to/topics.json \
  --max-skew-s 0.05
```

ROS1 `rosbag`, Zarr, NumPy, and Pillow are runtime dependencies. Writing occurs
in a temporary sibling directory; the requested output becomes visible only
after validation succeeds.
