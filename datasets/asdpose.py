"""ASDPose dataset for the raw-clips H5 format (skeleton-only SMM recognition).

Expected H5 keys (built by preprocessing/asdpose/build_h5.py):
    frames: (total_frames, 17, 3) float32  [x_norm, y_norm, score]
    seq_indptr: (N+1,) int64
    binary_label: (N,) uint8
    identifier: (N,) string
    split: (N,) uint8  (0=train, 1=test)
    train_row_indices / test_row_indices: (n_split,) int32
    metadata arrays (child_id, assessment_id, video_id, action_name, ...)

Each item returns:
    data_numpy: (3, T, 17, 1) float32
    label: int
    index: int   (index within THIS dataset; window-level in test sliding mode)
"""

import numpy as np
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None


class ASDPose(Dataset):
    """ASDPose clip/window dataset.

    - train: one item per clip, random temporal crop to `window_size`.
    - test (test_multi_windows=False): one item per clip, deterministic
      center/left/right crop.
    - test (test_multi_windows=True): each clip expands into sliding windows
      (stride `test_window_stride`, optional tail window) for clip-level
      score aggregation at evaluation.
    """

    def __init__(
        self,
        data_path,
        split="train",
        debug=False,
        window_size=200,
        # deterministic single-crop test mode
        test_crop="center",      # "center", "left", "right"
        pad_mode="repeat_last",  # "repeat_last" or "zero"
        # sliding-window test mode
        test_multi_windows=False,
        test_window_stride=50,
        test_include_tail=True,
    ):
        if h5py is None:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.data_path = data_path
        self.split = split
        self.debug = debug

        self.window_size = int(window_size) if window_size is not None else -1
        self.test_crop = test_crop
        self.pad_mode = pad_mode

        self.test_multi_windows = bool(test_multi_windows)
        self.test_window_stride = int(test_window_stride)
        self.test_include_tail = bool(test_include_tail)

        # HDF5 handles (lazy open per worker)
        self._h5 = None
        self._frames_ds = None

        # Clip-level metadata (per raw clip in this split)
        self.clip_row_indices = None
        self.seq_indptr = None
        self.clip_seq_start = None
        self.clip_seq_end = None
        self.clip_seq_len = None
        self.clip_label = None
        self.clip_identifier = None

        self.clip_child_id = None
        self.clip_assessment_id = None
        self.clip_video_id = None
        self.clip_action_name = None
        self.clip_event_start_frame_global = None
        self.clip_event_end_frame_global = None
        self.clip_fps = None

        # Item-level index (what __getitem__ sees)
        #   - train: 1 item per clip
        #   - test sliding: many items per clip
        self.item_clip_index = None             # shape (N_items,)
        self.item_window_start_local = None     # -1 for non-sliding items
        self.item_window_end_local_excl = None  # -1 for non-sliding items

        # Item-level labels / names (names must be unique for score aggregation)
        self.label = None
        self.sample_name = None

        self._load_metadata()

    # -------------------------------------------------
    # H5 utilities
    # -------------------------------------------------
    def _decode_str_arr(self, arr):
        out = []
        for x in arr:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8", errors="ignore"))
            else:
                out.append(str(x))
        return out

    def _ensure_h5_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
            self._frames_ds = self._h5["frames"]

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # -------------------------------------------------
    # Build clip metadata + item index
    # -------------------------------------------------
    def _load_metadata(self):
        with h5py.File(self.data_path, "r") as f:
            self.seq_indptr = np.asarray(f["seq_indptr"][:], dtype=np.int64)

            # split rows (global row indices)
            if self.split == "train":
                rows = np.asarray(f["train_row_indices"][:], dtype=np.int64)
            elif self.split == "test":
                rows = np.asarray(f["test_row_indices"][:], dtype=np.int64)
            else:
                raise ValueError("split must be 'train' or 'test'")

            if self.debug:
                rows = rows[: min(200, len(rows))]

            self.clip_row_indices = rows

            # clip frame ranges
            self.clip_seq_start = self.seq_indptr[rows]
            self.clip_seq_end = self.seq_indptr[rows + 1]
            self.clip_seq_len = (self.clip_seq_end - self.clip_seq_start).astype(np.int32)

            # clip labels + ids
            self.clip_label = np.asarray(f["binary_label"][rows], dtype=np.int64)
            self.clip_identifier = self._decode_str_arr(f["identifier"][rows])

            # clip metadata
            self.clip_child_id = np.asarray(f["child_id"][rows], dtype=np.int32)
            self.clip_assessment_id = np.asarray(f["assessment_id"][rows], dtype=np.int32)
            self.clip_video_id = np.asarray(f["video_id"][rows], dtype=np.int32)
            self.clip_action_name = self._decode_str_arr(f["action_name"][rows])
            self.clip_event_start_frame_global = np.asarray(f["event_start_frame_global"][rows], dtype=np.int32)
            self.clip_event_end_frame_global = np.asarray(f["event_end_frame_global"][rows], dtype=np.int32)
            self.clip_fps = np.asarray(f["fps"][rows], dtype=np.float32)

        # Build item-level index
        self._build_item_index()

        # Item-level labels
        self.label = self.clip_label[self.item_clip_index].astype(np.int64)

        # Item-level sample names
        # In test sliding mode names must be unique or the score dict would overwrite entries.
        names = []
        for i in range(len(self.item_clip_index)):
            ci = int(self.item_clip_index[i])
            base = self.clip_identifier[ci]
            ws = int(self.item_window_start_local[i])
            we = int(self.item_window_end_local_excl[i])

            if self.split == "test" and self.test_multi_windows and ws >= 0:
                names.append(f"{base}__ws{ws:05d}_we{we:05d}")
            else:
                names.append(base)
        self.sample_name = names

        self._len = len(self.item_clip_index)

        num_pos_clips = int((self.clip_label == 1).sum())
        num_neg_clips = int((self.clip_label == 0).sum())

        msg = (
            f"[ASDPose] split={self.split} "
            f"clips={len(self.clip_row_indices)} items={self._len} "
            f"label0={num_neg_clips} label1={num_pos_clips} "
            f"T(min/med/mean/max)=({int(self.clip_seq_len.min())}/"
            f"{int(np.median(self.clip_seq_len))}/{float(self.clip_seq_len.mean()):.2f}/"
            f"{int(self.clip_seq_len.max())})"
        )

        if self.split == "test" and self.test_multi_windows:
            avg_w = float(self._len) / max(1, len(self.clip_row_indices))
            msg += f"  sliding=True stride={self.test_window_stride} avg_windows_per_clip={avg_w:.2f}"

        print(msg)

    def _make_test_window_starts(self, T):
        """Deterministic sliding-window starts for a clip of length T."""
        L = int(self.window_size)

        if L <= 0:
            return [0]

        if T <= L:
            return [0]

        stride = max(1, int(self.test_window_stride))
        last_start = T - L

        starts = list(range(0, last_start + 1, stride))
        if self.test_include_tail and starts[-1] != last_start:
            starts.append(last_start)

        return starts

    def _build_item_index(self):
        """Build the dataset items exposed by __getitem__."""
        n_clip = len(self.clip_row_indices)

        # Default: one item per clip
        if not (self.split == "test" and self.test_multi_windows and self.window_size > 0):
            self.item_clip_index = np.arange(n_clip, dtype=np.int64)
            self.item_window_start_local = np.full((n_clip,), -1, dtype=np.int32)
            self.item_window_end_local_excl = np.full((n_clip,), -1, dtype=np.int32)
            return

        # Sliding-window test expansion
        item_clip_index = []
        item_ws = []
        item_we = []

        for ci in range(n_clip):
            T = int(self.clip_seq_len[ci])
            starts = self._make_test_window_starts(T)

            for ws in starts:
                we = min(ws + int(self.window_size), T)  # exclusive (before padding)
                item_clip_index.append(ci)
                item_ws.append(ws)
                item_we.append(we)

        self.item_clip_index = np.asarray(item_clip_index, dtype=np.int64)
        self.item_window_start_local = np.asarray(item_ws, dtype=np.int32)
        self.item_window_end_local_excl = np.asarray(item_we, dtype=np.int32)

    # -------------------------------------------------
    # Temporal processing
    # -------------------------------------------------
    def _pad_to_window(self, feat):
        """feat: (T, V, 3) -> padded to window_size if needed."""
        if self.window_size is None or self.window_size <= 0:
            return feat

        L = int(self.window_size)
        T = int(feat.shape[0])

        if T >= L:
            return feat[:L]

        pad = L - T
        if self.pad_mode == "zero":
            pad_feat = np.zeros((pad, feat.shape[1], feat.shape[2]), dtype=feat.dtype)
        else:
            last = feat[-1:, :, :]
            pad_feat = np.repeat(last, pad, axis=0)

        return np.concatenate([feat, pad_feat], axis=0)

    def _temporal_crop_pad_single(self, feat):
        """Single-window mode (train random crop / test deterministic crop).

        feat: (T, V, 3) -> (window_feat, ws, we_excl)
        """
        if self.window_size is None or self.window_size <= 0:
            return feat, 0, int(feat.shape[0])

        L = int(self.window_size)
        T = int(feat.shape[0])

        if T == L:
            return feat, 0, T

        # Crop
        if T > L:
            if self.split == "train":
                ws = int(np.random.randint(0, T - L + 1))
            else:
                if self.test_crop == "left":
                    ws = 0
                elif self.test_crop == "right":
                    ws = T - L
                else:
                    ws = (T - L) // 2
            we = ws + L
            return feat[ws:we], ws, we

        # Pad (T < L)
        ws = 0
        we = T
        return self._pad_to_window(feat), ws, we

    def _slice_window_by_start(self, feat, ws):
        """Sliding-window mode: deterministic window start ws.

        feat: (T, V, 3), ws in [0, T) -> (window_feat, ws, we_excl)
        """
        if self.window_size is None or self.window_size <= 0:
            return feat, 0, int(feat.shape[0])

        L = int(self.window_size)
        T = int(feat.shape[0])
        ws = int(max(0, ws))

        if T <= L:
            # whole clip + pad
            return self._pad_to_window(feat), 0, T

        we = min(ws + L, T)
        sub = feat[ws:we]
        if sub.shape[0] < L:
            sub = self._pad_to_window(sub)
        return sub, ws, we

    def _to_ctvm(self, feat):
        """feat: (T, V, 3) -> (C=3, T, V, M=1); channels: x, y, score."""
        T, V, C3 = feat.shape
        assert C3 == 3, f"Expected 3 channels [x, y, score], got {C3}"

        out = np.zeros((3, T, V, 1), dtype=np.float32)
        out[0, :, :, 0] = feat[:, :, 0]  # x
        out[1, :, :, 0] = feat[:, :, 1]  # y
        out[2, :, :, 0] = feat[:, :, 2]  # score
        return out

    # -------------------------------------------------
    # Dataset API
    # -------------------------------------------------
    def __len__(self):
        return self._len

    def __getitem__(self, index):
        index = int(index)
        ci = int(self.item_clip_index[index])  # parent clip index in this split

        s = int(self.clip_seq_start[ci])
        e = int(self.clip_seq_end[ci])

        # read raw clip frames (T, 17, 3)
        self._ensure_h5_open()
        feat = np.asarray(self._frames_ds[s:e], dtype=np.float32)

        # choose window
        if self.split == "test" and self.test_multi_windows and self.item_window_start_local[index] >= 0:
            ws = int(self.item_window_start_local[index])
            feat, _, _ = self._slice_window_by_start(feat, ws)
        else:
            feat, _, _ = self._temporal_crop_pad_single(feat)

        x = self._to_ctvm(feat)  # (3, T, V, 1)

        y = int(self.label[index])
        return x, y, index

    # -------------------------------------------------
    # Metadata for window -> clip aggregation at evaluation
    # -------------------------------------------------
    def get_meta(self, index):
        index = int(index)
        ci = int(self.item_clip_index[index])

        ws = int(self.item_window_start_local[index])
        we_excl = int(self.item_window_end_local_excl[index])

        # In non-sliding mode the window is not fixed at metadata build time
        # (train random crop especially). Use -1 as sentinel.
        if ws < 0:
            ws_out = -1
            we_out = -1
        else:
            ws_out = ws
            we_out = we_excl - 1  # inclusive

        return {
            # item-level info
            "dataset_index": index,
            "is_test_sliding_window": bool(self.split == "test" and self.test_multi_windows),
            "window_start_local": ws_out,
            "window_end_local": we_out,  # inclusive
            "window_size": int(self.window_size),

            # parent clip info
            "clip_index_in_split": ci,
            "identifier": self.clip_identifier[ci],
            "row_index_global": int(self.clip_row_indices[ci]),
            "child_id": int(self.clip_child_id[ci]),
            "assessment_id": int(self.clip_assessment_id[ci]),
            "video_id": int(self.clip_video_id[ci]),
            "action_name": self.clip_action_name[ci],
            "event_start_frame_global": int(self.clip_event_start_frame_global[ci]),
            "event_end_frame_global": int(self.clip_event_end_frame_global[ci]),
            "fps": float(self.clip_fps[ci]),
            "clip_len_raw": int(self.clip_seq_len[ci]),
            "label": int(self.clip_label[ci]),
        }
