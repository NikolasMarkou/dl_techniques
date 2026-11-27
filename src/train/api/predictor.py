"""Reload a trained run and predict with it: ``task.json``, ``load_model`` and ``Predictor``.

A run directory holds ``task.json`` beside its models: what a model needs to be used
without the config that trained it (the class names, the input geometry, and the
per-channel mean and standard deviation the training inputs were standardized with,
including the statistics of a folder or csv run's fit split, D-060). ``load_model``
reads it, imports the family's model code so its classes are registered, loads the
Keras file and returns a :class:`Predictor`, which applies the loader's own
preprocessing to a path, a PIL image or an array and returns class names and
probabilities.

Interface contract
------------------
``task.json`` (schema version 2), a JSON object. Its keys depend on its ``task`` (the record
carries the task word, so a reader dispatches on it and needs no second source). A classification
record has EXACTLY the keys and order of a version 1 record; only ``schema_version`` moved.
:data:`RECORD_KEYS` lists them per task. Every key of the task must be there, and no other, apart
from the OPTIONAL ``variant_overrides`` and ``init`` (:data:`OPTIONAL_RECORD_KEYS`; the ``init`` block, where a run's
starting weights came from, is checked key by key: :data:`INIT_KEYS`).

``schema_version``
    ``2``. Any other value is refused: a reader never guesses at a newer layout, and there is no
    reader for version 1 (D-004). A version 1 record gets one sentence naming the file and saying
    the run has to be trained again; ``predict``, ``export`` and ``resume`` all reach it
    (:func:`check_record`, :func:`refuse_old_record`).
``task``, ``model``, ``family``, ``variant``
    The task word (``image_classification``, ``image_segmentation`` or ``image_denoising``),
    ``"family/variant"``, and its two halves. A task this reader does not know is named and refused.
``class_names``
    One text per class, in the order of the model's outputs. **Limit, stated because it cannot
    be checked:** the reader verifies the COUNT (against the model's output width) and that no
    name repeats, never the ORDER. A record whose names are a permutation of the right ones (two
    classes swapped by hand or by a bug in whatever wrote it) loads and predicts, and reports the
    wrong name for those classes; no property of the model or of the record tells the two apart.
    Trust the file only as far as you trust its writer; ``export`` and a training run write it
    from the data they loaded.
``input_shape`` (classification), ``input_size`` (segmentation), ``channels``
    Classification: ``input_shape`` is ``[side, side, channels]`` of the model's input and
    ``channels`` repeats the last. Segmentation: ``input_size`` is ``[height, width]``, each above
    zero and not necessarily equal, and ``channels`` is the model's input channel count; there is
    no ``input_shape``, so the two cannot disagree.
``normalize``
    ``"standardize"`` or ``"unit"`` (what the run asked for).
``mean``, ``std``
    One number per channel: what the training inputs were standardized with, as the
    data loader reports it. ``unit`` is zeros and ones. A folder or csv run records the
    statistics of ONE pass over its fit split, not an ImageNet constant. **Bounds:**
    ``MEAN_RANGE`` (0 to 1) for each ``mean`` and ``STD_RANGE`` (1e-6 to 1) for each ``std``.
    The loader standardizes images that lie in ``[0, 1]``, so the mean of its pixels lies in
    ``[0, 1]`` and their standard deviation cannot pass 0.5 (``unit`` records 1); the loader
    itself refuses a folder or csv channel whose deviation is under 1e-6. A value outside these
    is not a statistic of such images, and the predictor would divide by it: a ``mean`` of 1e30
    turns every input into -1e30 and the probabilities into NaN, with no error. It is refused
    at load.
``eval_transform`` (classification only)
    ``{"kind", "source_channels"}``: how an image of any size becomes the model's
    input before standardization (:data:`EVAL_KINDS`, the evaluation pipelines of the
    loader), and the channel count the files are read with (``1`` for MNIST, whose
    single channel is repeated up to ``channels``).
``ignore_index`` (segmentation only)
    ``null``, or a whole number of 0 or more that is not a class index (outside
    ``0 .. len(class_names) - 1``): the mask value that marks a pixel as void. A mask file cannot
    hold a negative, and a class index would be a class the model cannot learn.
``mask_threshold`` (segmentation only)
    ``null``, or the finite number that turns a grey mask into two classes. It needs exactly two
    ``class_names`` and no ``ignore_index`` (D-014: an explicit number, two classes).
``output`` (segmentation only)
    ``"masks"`` or ``"logits"``: what the predictor returns for an image
    (:class:`~train.api.segmentation_predictor.SegmentationPredictor` reads it).

``channels``, ``size_multiple``, ``data_range``, ``output`` (denoising only)
    A denoising record has no ``class_names``, no ``normalize``, ``mean`` or ``std`` and no input size:
    the model's input is ``(None, None, channels)`` and its pixels lie in ``[0, 1]``, unstandardized.
    ``channels`` is 1 or 3. ``size_multiple`` is a positive int, the multiple each side of an image is
    reflect-padded to for a whole-image pass (the family's stride, 1 for a family with none).
    ``data_range`` is the text ``"[0,1]"`` and nothing else (``unit_domain_stamp``, the one definition the
    provenance gate of the legacy eval scripts reads from ``config.json``). ``output`` is ``"images"``
    (:data:`DENOISING_OUTPUTS`): the predictor returns the denoised image, raw, at each input's size.
    The zero image a loaded denoiser is checked with is ``input_shape_of``: ``DENOISING_PROBE_SIDE``
    rounded up to a multiple of ``size_multiple``, on each side.

``lookback``, ``horizon``, ``channels``, ``value_cols``, ``time_col``, ``scale_floor``, ``head``, ``quantile_levels``, ``point_index``, ``decode``, ``data_format``, ``data_sha256`` (forecasting only)
    A forecasting record has no ``class_names``, no ``normalize``, ``mean`` or ``std`` and no input size: the model takes a
    window ``(lookback, channels)`` in the series' own units and the statistics are those of each window's own context
    (:func:`train.common.forecast_data.context_stats_np`; ``scale_floor`` is its flatness threshold). ``lookback``, ``horizon``
    and ``channels`` are positive ints. ``head`` is ``point`` or ``quantile``; a quantile head has ``quantile_levels`` (a list
    of numbers in 0 to 1) and ``point_index`` (the level the point forecast is, the median), a point head has both null.
    ``decode`` is the name of the family's decode row (:data:`train.common.forecast_run.DECODE`). ``data_format`` is
    ``builtin`` (synthetic: ``value_cols`` and ``data_sha256`` null) or ``csv`` (``value_cols`` the ``channels`` column names
    the run read, ``data_sha256`` the 64 hex digits of the file's bytes, ``time_col`` the name of the time column the run checked
    for order, or null: ``predict`` of a csv checks the order with the same function the loader uses, and skips it for null;
    ``time_col`` is null for ``builtin``). ``load_model`` refuses a record whose lookback or channels are not the ones the model
    was BUILT for (:func:`train.api.forecast_predictor.built_window`, every family, D-042) and one whose decode row, head,
    levels, median or forecast shape is not the model's (:class:`~train.api.forecast_predictor.ForecastPredictor`). A resume
    compares :data:`DATA_KEYS` and nothing about the head. The zero window a loaded forecaster is checked with is
    ``(1, lookback, channels)``. ``scale_floor`` is NOT checked against anything: the model does not carry it, so a hand-edited
    value cannot be detected and changes every forecast (the flatness threshold of the standardization).

``input_size``, ``channels``, ``normalize``, ``output`` (depth estimation)
    A depth record has no ``class_names``, ``mean`` or ``std``: ``input_size`` is ``[height, width]`` the model was BUILT at
    (positive ints), ``channels`` is 3, ``normalize`` is ``"imagenet"`` (:data:`DEPTH_NORMALIZE`: the ImageNet mean and
    standard deviation of the official DINOv2 input, one function, ``train.common.megadepth.standardize_imagenet``, in
    training and in ``predict``) and ``output`` is ``"relative_disparity"`` (:data:`DEPTH_OUTPUTS`: larger is nearer, defined
    up to a scale and a shift per image). ``load_model`` refuses a record whose size or channels are not the ones the model
    was built at (:func:`train.api.depth_predictor.check_built_size`, the ``MODEL_CHECKS`` row), before the model is called:
    the model takes no other size, and a size that is not a multiple of the patch would pass a zero-input call. A resume
    compares ``input_size``, ``channels`` and ``normalize``.

``input_size``, ``channels``, ``normalize``, ``mean``, ``std``, ``patch_size``, ``tokenizer``, ``output`` (pretrain)
    A pretrain record has no ``class_names``: ``input_size`` is ``[side, side]`` the model was BUILT at (the pretrain
    streams make square images), ``channels`` is 3, ``normalize`` is ``"standardize"`` (:data:`PRETRAIN_NORMALIZE`: the
    word image_classification records for the same dataset) with ``mean`` and ``std`` the statistics the streams
    standardize with (``train.common.image_streams.stream_statistics``), ``patch_size`` is the side of a patch (a
    positive int dividing ``input_size``), ``tokenizer`` is the 64 lowercase hex digits of the sha256 of a BEiT run's
    tokenizer file (``null`` for a family with none) and ``output`` is ``"patch_embeddings"`` (:data:`PRETRAIN_OUTPUTS`:
    the backbone's patch-token embeddings, ``N = (side / patch_size)^2`` rows per image, a class-token row dropped).
    ``load_model`` refuses a record whose size or channels are not the backbone's before the model is called
    (:func:`train.api.pretrain_predictor.check_built_size`), and one whose patch is not the model's by the rows of its
    output on a zero image (:func:`train.api.pretrain_predictor.check_output`). A resume compares the geometry, the
    scaling, the patch and the tokenizer.

``input_size``, ``channels``, ``normalize``, ``mean``, ``std``, ``patch_size``, ``num_embeddings``, ``output`` (beit_tokenizer)
    A tokenizer record has the pretrain record's geometry and scaling (the two tasks read the same image streams: ``input_size`` is
    ``[side, side]`` the model was BUILT at, ``channels`` 3, ``normalize`` ``"standardize"`` with the streams' ``mean`` and ``std``,
    ``patch_size`` a positive int dividing the side) and no ``tokenizer`` file hash. ``patch_size`` is the side of the cell one code
    describes, the encoder's total stride, so the code grid is ``input_size / patch_size`` on a side; ``num_embeddings`` is the codebook
    size, a positive int; ``output`` is ``"code_ids"`` (:data:`TOKENIZER_OUTPUTS`: the int32 grid of codes, row-major).
    ``load_model`` refuses, before the model is called, a record whose size, patch or codebook is not the model's
    (:func:`train.api.tokenizer_predictor.check_built_model`, which also proves the code grid by running the model on a zero image), a
    variant the model is not (``ModelSpec.observe``) and a file that is no ``VQVAERotationTrick``; the model's reconstruction of the zero
    image is held to the record's size (:func:`train.api.tokenizer_predictor.check_output`). A resume compares ``input_size``,
    ``channels``, ``normalize``, ``mean``, ``std`` and ``patch_size``: the codebook is the model's and ``output`` is how the predictor answers.

``seq_len``, ``encoding``, ``n_vocab``, ``vocab_size``, ``eot_id``, ``pad_id``, ``tokenizer``, ``tokenizer_files``, ``output`` (language model)
    A language-model record has no ``class_names``, input size or statistics: the model takes ``seq_len`` token ids and returns the
    next-token logits ``(seq_len, vocab_size)`` (``output`` is ``"next_token_logits"``, :data:`LANGUAGE_OUTPUTS`). ``encoding`` is the
    tokenizer (:data:`train.api.text_tokenizer.TOKENIZERS`), ``n_vocab`` its own ids, ``vocab_size`` the model's, ``eot_id`` and ``pad_id``
    ids below it; ``tokenizer`` is the digest of its :class:`~train.api.text_tokenizer.TokenizerSpec` and ``tokenizer_files`` the sha256
    of each BPE file (``{}`` for ``ascii95``). ``check_record`` checks the formats and that the ids lie in the vocabulary, and, for a record
    that names the digest this code pins, that ``eot_id`` and ``pad_id`` (``cls_id`` too) are the ones that digest is made of; ``predict``
    checks the digest against the one this code pins, so a record of another tokenizer is still a record (``export`` and ``init``
    read it). ``load_model`` refuses a ``seq_len`` or ``vocab_size`` that is not the model's, by the zero document's logits. A resume
    compares :data:`DATA_KEYS`.

``cls_id``, ``class_names``, ``layout``, ``output`` (text classification)
    A text-classification record has the language model's keys up to ``tokenizer_files`` and then ``cls_id`` (the id every row begins with:
    an id below ``vocab_size`` other than ``pad_id``, which the padding mask would hide), ``class_names`` (the label names in label order,
    as an image classifier's), ``layout`` (:data:`TEXT_LAYOUTS`: ``cls_head_pad``, a row is ``[CLS]``, the first ``seq_len - 1`` ids of the
    document and pad ids up to ``seq_len``) and ``output`` (:data:`TEXT_CLASSIFICATION_OUTPUTS`: ``logits``, one per class name).
    ``load_model`` refuses a ``seq_len`` or a class count that is not the model's, by the zero document's logits. A resume compares
    :data:`DATA_KEYS`: the language model's four and ``class_names``.

``tokenizer_dir`` (OPTIONAL, a language-model or text-classification record)
    The ABSOLUTE directory the run read its BPE files from, written ONLY when the run set ``data.tokenizer_dir`` (null: the pinned copy,
    no key). ``predict`` opens the tokenizer from it, after the hashes of the files are checked (D-077); a file of another hash is refused
    by name, and a directory that is gone is replaced by the pinned copy, hashed the same way (D-085).

``variant_overrides`` (OPTIONAL)
    ``{table key: value}``: the keys of the variant's table (``MODEL_VARIANTS[variant]``) that the
    run's resolved ``model_args`` set to another value, so the model is the variant WITH these
    changes. Written ONLY when there is at least one (a default run's ``task.json`` is
    byte-identical to one written before the key existed); a record may omit it, and an empty or
    malformed one is refused. :func:`check_model_against_record` compares the model with the
    variant's table updated by it, so an override is honoured and a wrong variant or an override the
    model does not have is still refused. ``export`` carries it into ``<out>.task.json``.

``build_task_record(...)`` / ``build_segmentation_record(...)`` / ``build_denoising_record(...)`` / ``build_forecast_record(...)`` / ``build_depth_record(...)`` / ``build_pretrain_record(...)`` / ``build_language_model_record(...)`` / ``build_text_classification_record(...)`` / ``build_tokenizer_record(...)``
    :return: the record for a finished data load (not checked; see their docstrings). The first
        writes a classification record, the second a segmentation record, the third a denoising one, the fourth a forecasting
        one, the fifth a depth one, the sixth a pretrain one, the seventh a language-model one, the eighth a text-classification one,
        the ninth a tokenizer one.

``check_record(record, source)``
    :param record: a parsed ``task.json``.
    :param source: its path, for the messages.
    :raises ConfigError: for a wrong ``schema_version`` (an older one in the sentence described
        above), a ``task`` this reader does not know, a missing or unknown key for that task, a
        value of the wrong kind (``variant_overrides`` included: a non-empty object of text keys
        and JSON values), repeated class names, a ``model`` that is not ``family/variant``, a
        ``mean`` or ``std`` of the wrong length, a ``std`` that is not positive, a ``mean`` or
        ``std`` outside the bounds above, ``normalize`` of ``unit`` with a ``mean`` that is not
        zeros or a ``std`` that is not ones, and the task's own keys: a classification record's
        ``input_shape`` that is not square or an ``eval_transform`` whose kind is not in
        :data:`EVAL_KINDS`; a segmentation record's ``input_size``, ``channels``, ``ignore_index``,
        ``mask_threshold`` (with its two-class and no-void conditions) or ``output``; a denoising
        record's ``channels``, ``size_multiple``, ``data_range`` or ``output``.

``refuse_old_record(path, prefix)``
    :param path: a ``task.json`` on disk.
    :param prefix: how the message begins (``"Cannot resume <dir>"``).
    :raises ConfigError: when the file is JSON whose ``schema_version`` is a whole number below
        :data:`SCHEMA_VERSION`; returns for a missing file, a file that is not JSON (the reader
        names that) and any other version (the reader refuses those). One sentence naming the
        file and the retraining. ``resume`` calls it before it loads the data.

``record_differences(earlier, now)``
    :param earlier: a checked record (a run's ``task.json``).
    :param now: a record built from the data loaded now (:func:`build_task_record` or
        :func:`build_segmentation_record`).
    :return: one ``"<key>: <earlier> -> <now>"`` line per key of ``DATA_KEYS[task]`` that
        differs: for classification the class names (as a list, in order), the geometry, the
        normalization and the per-channel ``mean`` and ``std`` (within a relative ``1e-5``); for
        segmentation the class names, ``input_size``, the normalization, ``mean`` and ``std``,
        ``ignore_index`` and ``mask_threshold``; for denoising ``channels`` and ``data_range``
        (not the held-out sets: D-027). ``output`` is how the predictor answers, not what
        the run trained on, and is not compared. Records of two tasks differ in ``task`` alone.
        A resume compares with this before it trains on the data (D-069).

``load_model(path)``
    :param path: a run directory (its ``best_model.keras``) or a ``.keras`` file. A
        ``.keras`` file reads the first record that exists of ``<file>.task.json`` (what
        ``export`` writes beside its output), ``<stem>.task.json``, and ``task.json`` in its
        directory.
    :return: a :class:`Predictor` for an ``image_classification`` record, a
        :class:`~train.api.segmentation_predictor.SegmentationPredictor` for an
        ``image_segmentation`` one (dispatch on ``record["task"]``). Compiles nothing. The model is
        called once on zeros, so a model whose output width (classification) or output map
        (segmentation: ``input_size`` and the class count; denoising: ONE tensor of the probe image's
        size and the record's channels) does not match the record is refused here. A ``image_denoising``
        record gives a :class:`~train.api.denoising_predictor.DenoisingPredictor`, a ``time_series_forecasting`` one a
        :class:`~train.api.forecast_predictor.ForecastPredictor`, a ``depth_estimation`` one a
        :class:`~train.api.depth_predictor.DepthPredictor`, a ``pretrain`` one a
        :class:`~train.api.pretrain_predictor.PretrainPredictor`, a ``language_model_pretraining`` one a
        :class:`~train.api.text_predictor.LanguageModelPredictor`, a ``text_classification`` one a
        :class:`~train.api.text_predictor.TextClassifierPredictor`, a ``beit_tokenizer`` one a
        :class:`~train.api.tokenizer_predictor.TokenizerPredictor`.
    :raises ConfigError: for a missing path, ``task.json`` or model file, a model file that
        is not an intact ``.keras`` archive (one cut short by a kill, or a zip with no
        ``config.json`` or no weights, is named, not a Keras traceback), a bad record, a
        ``task`` this reader does not know, a family that is not registered or does not train the
        record's task, a family or variant the model is not (:func:`check_model_against_record`),
        an ``input_shape`` / ``input_size`` or ``channels`` the model does not take, or a
        class-count mismatch. All of them are raised by ``load_model``, none waits for the first
        prediction.

``record_beside(path)`` / ``input_shape_of(record)``
    :param path: what ``load_model`` takes.
    :return: the checked record of that run or file, without loading the model (the command line
        reads the task before it loads anything); and the model input a record names,
        ``(height, width, channels)``: ``input_shape`` for classification, ``input_size`` and
        ``channels`` for segmentation, the probe side (see above) twice and ``channels`` for denoising;
        ``(lookback, channels)`` (rank 2) for forecasting.

``EXPORT_RULES[task]`` (an :class:`ExportRule`)
    What ``export`` asks the task: ``shape_of(record)`` is the shape the command line traces (the record's input
    shape; a denoiser's is ``(None, None, channels)``, a free height and width), ``check(record, format, shape)``
    raises a :class:`ConfigError` for a cell of the task's export table that is refused (classification, denoising,
    forecasting and pretrain only, :data:`CLASSIFICATION_EXPORT_REFUSALS`,
    :data:`train.api.denoising_predictor.EXPORT_REFUSALS`,
    :data:`train.api.forecast_predictor.EXPORT_REFUSALS`, :data:`train.api.pretrain_predictor.EXPORT_REFUSALS`) and,
    for forecasting, depth, pretrain, the two text tasks and the tokenizer, for a traced shape that is not the record's (a depth model exports
    every format at its trained size only, :func:`train.api.depth_predictor.check_export`, and so do a pretrain model,
    :func:`train.api.pretrain_predictor.check_export`, a language model or a text classifier, :func:`train.api.text_predictor.check_export`,
    and a tokenizer, :func:`train.api.tokenizer_predictor.check_export`, which refuses no cell of the table: all four formats were measured), and
    ``note(record)`` is the sentence logged beside an export.
    ``served`` is ``None`` for the other image tasks (the file is the model), for forecasting the layer that
    standardizes a raw window, runs the model, decodes and de-standardizes
    (:class:`train.common.forecast_run.ServedForecaster`), and for pretrain a functional model over the run's backbone
    that returns the patch-token embeddings (:func:`train.api.pretrain_predictor.served`), and for a language model or a text classifier the
    model with ``training=False`` (:func:`train.api.text_predictor.served`: a rule with no served function is written with no tflite check
    at all), and for a tokenizer the layer that returns the int32 code grid (:class:`train.common.tokenizer_data.TokenizerCodes`);
    the rank of ``shape_of(record)`` is the rank ``export`` accepts of ``input_shape`` (3 for images, 2 for a forecaster, 1 for
    a text model). ``sample`` is the two text tasks' check batch of seeded ids (:func:`train.api.text_predictor.sample_ids`) and
    ``None`` for every other task. A row in every table of :data:`TASK_TABLES`, this one included.

``Predictor.predict(x, top_k=None, threshold=None)``
    :param x: one image or several. A single image is a path (``str`` or ``Path``), a
        ``PIL.Image.Image``, or a 2-D or 3-D ``numpy`` array; several are a list or tuple
        of those, or a 4-D array. A ``uint8`` array is divided by 255; a floating array
        must already lie in ``[0, 1]`` and is refused otherwise; any other dtype is
        refused. A 4-channel array is RGBA and loses its alpha channel, as a PIL image does. A path is read with ``channels`` of ``eval_transform.source_channels``.
        **Grey models** (``source_channels`` 1): a colour array or PIL image is made grey on its
        8-bit pixels with the loader's rule, ``(9797 R + 19234 G + 3737 B) >> 15``
        (:func:`train.common.classification_data.rgb8_to_grey8`; libpng's integer weights as
        ``tf.io.decode_image(channels=1)`` applies them to a PNG, equal to it on 65,536 random
        colours), and a floating array is rounded to its 8-bit values first, as storing it as a
        PNG would. So a PNG's pixels (RGB, RGBA, palette) come out as the loader's within 1e-5
        (measured: identical). **Measured limit, JPEG:** a JPEG given as a PATH is decoded by the
        loader's own call and is exact; a JPEG already decoded to an array or PIL image is not,
        because the loader's grey is the JPEG's Y plane, which RGB output no longer holds, and
        PIL's and TensorFlow's JPEG decoders differ besides. Against the loader's decode of the
        same file, in levels of 255: random colour noise at quality 75, worst pixel 14, mean 0.56;
        at quality 95, 13 and 1.15; smooth images, worst 1 to 2, mean 0.5 to 0.6 (about 0.002 of
        the scale). That is above 1e-5 and cannot be brought under it from decoded pixels; pass the
        file's path when the exact value matters. ``tf.image.rgb_to_grayscale`` and PIL's ``convert("L")`` (the earlier
        calls) differ from the loader on PNG pixels too: one level in a quarter and in a half of
        the pixels.
        **16-bit sources, by PIL mode:** ``I;16`` (and ``I;16L``, ``I;16B``, ``I;16N``) is
        converted as the decoder reads the file, each sample's high byte (``value >> 8``, equal
        to the loader on the same file), for one and three channels. ``I`` and ``F`` (32-bit
        integers and floats) are REFUSED with a :class:`ConfigError` naming the mode: the loader
        reads no file of that depth. A 16-bit RGB or RGBA PNG opens in PIL as 8-bit ``RGB`` or
        ``RGBA``, which cannot be told from an 8-bit file, so it is neither refused nor exact for
        a grey model: the decoder makes the grey from the 16-bit samples before it strips them, and
        the 8-bit pixels no longer hold that. **Measured limit:** at most 1 level of 255 against the loader's
        decode of the same file (a path is exact; a three-channel model is exact).
    :param top_k: keep the ``top_k`` best classes of each image; ``None`` keeps all.
    :param threshold: drop classes whose probability is below it; ``None`` drops none.
    :return: ``{"labels", "scores"}``. For a single image both are flat lists, best class
        first (ties keep class order); for several images, one such list per image.
        ``scores`` are softmax probabilities of the model's logits, as plain floats.
    :raises ConfigError: for an unreadable file, an unsupported or empty array, an empty list, a
        ``top_k`` that is not a positive int, a ``threshold`` outside ``[0, 1]``, or a
        model whose output width is not the number of class names.

``Predictor.model``
    The loaded Keras model.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); every
``keras``, ``tensorflow`` and ``train.common`` name is imported inside a function.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from . import registry
from .args import is_count
from .depth_args import TASK as DEPTH
from .errors import ConfigError
from .pretrain_args import TASK as PRETRAIN
from .recipes import ET_CLASSIFIER
from .run_dir import TASK_FILE, TASK_SUFFIX, check_model_file
from .text_args import CLASSIFICATION_TASK as TEXT_CLASSIFICATION, TASK as LANGUAGE_MODEL
from .dino_args import TASK as DINO
from .text_tokenizer import ASCII95, TOKENIZERS, tokenizer_spec
from .tokenizer_args import TASK as BEIT_TOKENIZER

# ---------------------------------------------------------------------

# DECISION plan-2026-10-31T180033-2faf890a/D-073: a record has the keys of ITS task and no others, and
# version 1 is not read. Do NOT make one flat record with every key and a null for the other task's
# (a key nothing reads is a lie in the file), do NOT keep ``input_shape`` beside ``input_size`` in a
# segmentation record (one fact held twice, and a square-only check on one of them), and do NOT add a
# version 1 reader "for the old runs": D-004 priced that, every old run directory loses predict,
# export and resume, and the standing rule is no legacy loaders.
#: The only ``schema_version`` this module reads and writes. Version 1 (classification only, one
#: layout) is not read: the number moved when segmentation records were added, a v1 run is
#: refused at predict, export and resume, and there is no converter (D-004, the standing rule
#: against legacy readers).
SCHEMA_VERSION = 2

CLASSIFICATION = "image_classification"
SEGMENTATION = "image_segmentation"
DENOISING = "image_denoising"
FORECASTING = "time_series_forecasting"

#: The task :class:`Predictor` serves; ``load_model`` returns a
#: :class:`~train.api.segmentation_predictor.SegmentationPredictor` for a segmentation record, so
#: a record of any other task is refused by the class that does not serve it.
SERVED_TASK = CLASSIFICATION

#: The keys the two text tasks' records share, in the order they are written: the window and the tokenizer's facts.
_TEXT_KEYS = (
    "schema_version", "task", "model", "family", "variant", "seq_len", "encoding", "n_vocab", "vocab_size", "eot_id", "pad_id",
    "tokenizer", "tokenizer_files",
)

#: The keys of a record per task, in the order they are written. The classification tuple is the
#: version 1 tuple: byte-identical output apart from ``schema_version`` is a standing requirement.
RECORD_KEYS: Dict[str, Tuple[str, ...]] = {
    CLASSIFICATION: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "class_names",
        "input_shape",
        "channels",
        "normalize",
        "mean",
        "std",
        "eval_transform",
    ),
    SEGMENTATION: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "class_names",
        "input_size",
        "channels",
        "normalize",
        "mean",
        "std",
        "ignore_index",
        "mask_threshold",
        "output",
    ),
    DENOISING: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "channels",
        "size_multiple",
        "data_range",
        "output",
    ),
    FORECASTING: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "lookback",
        "horizon",
        "channels",
        "value_cols",
        "time_col",
        "scale_floor",
        "head",
        "quantile_levels",
        "point_index",
        "decode",
        "data_format",
        "data_sha256",
    ),
    DEPTH: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "input_size",
        "channels",
        "normalize",
        "output",
    ),
    PRETRAIN: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "input_size",
        "channels",
        "normalize",
        "mean",
        "std",
        "patch_size",
        "tokenizer",
        "output",
    ),
    LANGUAGE_MODEL: (*_TEXT_KEYS, "output"),
    # The language model's keys, then the classifier's own: the id its rows begin with, the label names (the model's outputs, in
    # label order), how a row is made from a document and what the model returns.
    TEXT_CLASSIFICATION: (*_TEXT_KEYS, "cls_id", "class_names", "layout", "output"),
    # The pretrain record's geometry and scaling, the codebook size in place of the tokenizer file's hash (D-044).
    BEIT_TOKENIZER: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "input_size",
        "channels",
        "normalize",
        "mean",
        "std",
        "patch_size",
        "num_embeddings",
        "output",
    ),
    # The pretrain record's geometry and scaling (the two tasks read the same image streams), then what makes a view set and a feature: the
    # local crops (the model takes ``2 + n_local_crops`` views of one image), the projection head's width and the embedding's width.
    DINO: (
        "schema_version",
        "task",
        "model",
        "family",
        "variant",
        "input_size",
        "channels",
        "normalize",
        "mean",
        "std",
        "patch_size",
        "n_local_crops",
        "dino_out_dim",
        "embedding_dim",
        "output",
    ),
}

#: What ``output`` may say in a segmentation record.
SEGMENTATION_OUTPUTS = ("masks", "logits")

#: What ``output`` may say in a denoising record: ``images``, the denoised image at each input's size.
DENOISING_OUTPUTS = ("images",)

#: The side of the zero image a loaded denoiser is called on to check its output, rounded UP to a multiple of
#: the record's ``size_multiple`` (a U-Net refuses a side off its stride; the model's own input is flexible).
DENOISING_PROBE_SIDE = 32

#: What ``normalize`` says in a depth record: the ImageNet statistics of the official DINOv2 input; what ``output`` may say (the
#: model's map, larger is nearer, defined up to a scale and a shift per image); the channels its model takes (RGB).
DEPTH_NORMALIZE = "imagenet"
DEPTH_OUTPUTS = ("relative_disparity",)
DEPTH_CHANNELS = 3

#: What ``normalize`` says in a pretrain record (the streams always standardize, with ``mean`` and ``std``: the word
#: image_classification records for the same dataset); what ``output`` may say (the backbone's patch-token embeddings);
#: the channels its model takes (the pretrain streams are RGB).
PRETRAIN_NORMALIZE = "standardize"
PRETRAIN_OUTPUTS = ("patch_embeddings",)
PRETRAIN_CHANNELS = 3

#: What ``output`` may say in a ``beit_tokenizer`` record: the int32 code of every patch cell, row-major. The task reads the pretrain
#: streams, so its ``normalize`` word, its channels and its geometry rules are the pretrain record's (:data:`PRETRAIN_NORMALIZE`,
#: :data:`PRETRAIN_CHANNELS`, ``_pretrain_geometry``).
TOKENIZER_OUTPUTS = ("code_ids",)

#: What ``output`` may say in a ``dino`` record: the student's CLS feature (the post-final-norm token, ``embedding_dim`` wide), never the
#: projection head's output. The task reads the pretrain streams, so its ``normalize`` word, channels and geometry rules are the pretrain record's.
DINO_OUTPUTS = ("cls_embeddings",)

#: What ``output`` may say in a language-model record: the logits over the vocabulary at every position of the window (the row of
#: position ``i`` is the distribution of the id after the ``i``-th).
LANGUAGE_OUTPUTS = ("next_token_logits",)

#: What ``output`` and ``layout`` may say in a text-classification record: the raw class logits, and how a row is made from a document
#: (``[CLS]``, the first ``seq_len - 1`` ids, pad ids up to ``seq_len``: ``train.common.text_data.encode_rows``).
TEXT_CLASSIFICATION_OUTPUTS = ("logits",)
TEXT_LAYOUTS = ("cls_head_pad",)

#: What ``head`` may say in a forecasting record, and the sources ``data_format`` may name (the ``data.format`` values of the task).
FORECAST_HEADS = ("point", "quantile")
FORECAST_FORMATS = ("builtin", "csv")

#: Keys a record MAY carry (written only when they hold something); every other key is required.
OPTIONAL_RECORD_KEYS = ("variant_overrides", "init", "tokenizer_dir")

#: The keys of the optional ``init`` block (where a run's starting weights came from: ``train.api.init_from``), in the
#: order they are written; ``transfer`` holds :data:`INIT_TRANSFER_KEYS`, each ``resampled`` entry
#: :data:`INIT_RESAMPLED_KEYS`; ``relation`` is one of :data:`INIT_RELATIONS`; ``split`` holds :data:`INIT_SPLIT_KEYS`.
INIT_KEYS = ("source", "file", "sha256", "task", "family", "variant", "geometry", "relation", "pinned", "transfer",
             "split")
INIT_TRANSFER_KEYS = ("copied", "resampled", "reset", "grown", "dropped")
INIT_RESAMPLED_KEYS = ("path", "rule", "from", "to")
INIT_RELATIONS = ("same_task", "cross_task")
#: The ``init.split`` block (``init_from.split_block``): how the source's training data meets this run's held-out data.
INIT_SPLIT_KEYS = ("relation", "unit", "n_select_overlap", "n_test_overlap", "reason", "keys")
INIT_SPLIT_RELATIONS = ("same_split", "different_data", "overlap", "unknown")

#: The evaluation transforms this predictor reproduces, by the loader code each one is:
#: ``resize`` is ``train.common.classification_data.resizer`` (the in-memory datasets,
#: folder and csv data); ``resize_centre_crop`` is
#: ``dl_techniques.datasets.vision.crops.resize_and_centre_crop`` (Imagenette);
#: ``imagenet_centre_crop`` is ``dl_techniques.datasets.vision.imagenet2012.center_crop``
#: (ImageNet's antialiased crop).
EVAL_KINDS = ("resize", "resize_centre_crop", "imagenet_centre_crop")

#: The evaluation transform each data source records.
# DECISION plan-2026-10-30T132432-a865fa08/D-062: ImageNet records ``imagenet_centre_crop`` and
# the predictor calls the loader's own ``center_crop`` for it. Do NOT record ``resize`` for
# ImageNet and do NOT approximate its crop with Imagenette's (an unrounded shorter side and no
# antialias): a prediction through another crop looks fine and is wrong. Each kind names the
# one loader function the predictor calls, so a new kind is a new row here and a branch in
# ``Predictor._shaped``, never a refusal at read.
EVAL_KIND_OF_DATA = {
    "mnist": "resize",
    "cifar10": "resize",
    "cifar100": "resize",
    "folder": "resize",
    "csv": "resize",
    "imagenette": "resize_centre_crop",
    "imagenet": "imagenet_centre_crop",
}

#: Images turned into tensors and run through the model at a time.
CHUNK = 32

_NORMALIZE = ("standardize", "unit")

#: The range of a record's per-channel ``mean`` and ``std``. The loader's images lie in [0, 1], so
#: their mean does too and their deviation cannot pass 0.5 (``unit`` records exactly 1); 1e-6 is
#: the loader's own floor for a folder or csv channel (``classification_data`` refuses less).
#: A value outside is no statistic of such images and the predictor divides by it.
MEAN_RANGE = (0.0, 1.0)
STD_RANGE = (1e-6, 1.0)

#: The PIL modes that already are grey (read as ``L``); every other mode is read as RGB.
GREY_MODES = ("L", "LA", "1")

#: The 16-bit grey PIL modes (a 16-bit grey PNG opens as ``I;16``; the others differ in byte
#: order). Read as the decoder reads the same file: each sample stripped to its high byte.
GREY16_MODES = ("I;16", "I;16L", "I;16B", "I;16N")

#: The PIL modes of 32-bit integers and floats. REFUSED: the loader reads no file of that depth,
#: so there is no decoder rule to reproduce.
WIDE_MODES = ("I", "F")


def _record_of(
    task: str, values: Mapping[str, Any], variant_overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Return the record of ``task``: ``values`` in the order of ``RECORD_KEYS[task]``, plus the optional keys that hold something."""
    record = {key: values[key] for key in RECORD_KEYS[task]}
    if values.get("tokenizer_dir") is not None:      # only a text record's values hold the key; absolute, so `predict` reads it from any directory
        record["tokenizer_dir"] = str(Path(values["tokenizer_dir"]).resolve())
    if variant_overrides:
        record["variant_overrides"] = _plain(dict(variant_overrides))
    return record


def _statistics(mean: Any, std: Any) -> Dict[str, Any]:
    return {"mean": [float(value) for value in mean], "std": [float(value) for value in std]}


def build_task_record(
    *,
    task: str,
    model: str,
    family: str,
    variant: str,
    class_names: Sequence[Any],
    input_shape: Sequence[int],
    normalize: str,
    mean: Any,
    std: Any,
    data_kind: str,
    source_channels: int,
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a classification run; see the module contract.

    :param task: written as given (the classification task passes ``image_classification``);
        the LAYOUT is the classification one whatever the word.
    :param variant_overrides: the output of :func:`overrides_of_variant`; the record gets the
        ``variant_overrides`` key only when this is not empty.

    :param data_kind: a key of :data:`EVAL_KIND_OF_DATA`: the dataset name, or ``folder``
        or ``csv``.
    :param source_channels: the channels the files are read with.
    :return: the record. It is NOT checked: ImageNet's antialiased centre crop is recorded
        truthfully; every kind it can hold is one :func:`check_record` accepts.
    """
    shape = [int(side) for side in input_shape]
    return _record_of(
        CLASSIFICATION,
        {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "model": model,
            "family": family,
            "variant": variant,
            "class_names": [str(name) for name in class_names],
            "input_shape": shape,
            "channels": shape[2],
            "normalize": normalize,
            **_statistics(mean, std),
            "eval_transform": {
                "kind": EVAL_KIND_OF_DATA[data_kind],
                "source_channels": int(source_channels),
            },
        },
        variant_overrides,
    )


def build_segmentation_record(
    *,
    model: str,
    family: str,
    variant: str,
    class_names: Sequence[Any],
    input_size: Sequence[int],
    channels: int,
    normalize: str,
    mean: Any,
    std: Any,
    ignore_index: Optional[int] = None,
    mask_threshold: Optional[float] = None,
    output: str = "masks",
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a segmentation run; see the module contract.

    :param input_size: ``(height, width)`` of the model's input, not necessarily equal.
    :param channels: the model's input channels.
    :param ignore_index: the void value of the masks, or ``None``.
    :param mask_threshold: the grey-mask threshold of a two-class run, or ``None``.
    :param output: one of :data:`SEGMENTATION_OUTPUTS`.
    :return: the record. It is NOT checked (:func:`check_record` does that).
    """
    return _record_of(
        SEGMENTATION,
        {
            "schema_version": SCHEMA_VERSION,
            "task": SEGMENTATION,
            "model": model,
            "family": family,
            "variant": variant,
            "class_names": [str(name) for name in class_names],
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": normalize,
            **_statistics(mean, std),
            "ignore_index": ignore_index,
            "mask_threshold": mask_threshold,
            "output": output,
        },
        variant_overrides,
    )


def unit_domain_stamp() -> str:
    """Return the text ``data_range`` holds for a denoiser trained on ``[0, 1]`` pixels (``"[0,1]"``).

    Interface contract: no parameters; returns the one definition the provenance gate of the legacy readers
    compares with (``dl_techniques.utils.denoiser_provenance.UNIT_DOMAIN_STAMP``), so a denoising record and
    ``config.json`` cannot disagree with it. The import is lazy and light (no TensorFlow).
    """
    from dl_techniques.utils.denoiser_provenance import UNIT_DOMAIN_STAMP

    return UNIT_DOMAIN_STAMP


def build_denoising_record(
    *,
    model: str,
    family: str,
    variant: str,
    channels: int,
    size_multiple: int,
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a denoising run; see the module contract.

    :param channels: the model's input and output channels (1 or 3).
    :param size_multiple: the multiple each side of an image is padded to for a whole-image pass (the
        family's stride; 1 for a family with none).
    :return: the record; ``data_range`` is the unit-domain stamp and ``output`` is ``images``. It is NOT checked
        (:func:`check_record` does that).
    """
    return _record_of(
        DENOISING,
        {
            "schema_version": SCHEMA_VERSION,
            "task": DENOISING,
            "model": model,
            "family": family,
            "variant": variant,
            "channels": int(channels),
            "size_multiple": int(size_multiple),
            "data_range": unit_domain_stamp(),
            "output": DENOISING_OUTPUTS[0],
        },
        variant_overrides,
    )


def build_forecast_record(
    *,
    model: str,
    family: str,
    variant: str,
    lookback: int,
    horizon: int,
    channels: int,
    value_cols: Optional[Sequence[str]],
    time_col: Optional[str],
    scale_floor: float,
    decode: str,
    quantile_levels: Optional[Sequence[float]],
    point_index: Optional[int],
    data_format: str,
    data_sha256: Optional[str],
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a forecasting run; see the module contract.

    :param quantile_levels, point_index: ``None`` for a point head; a quantile head gives both, and ``head`` follows them.
    :param decode: the name of the model's decode row (``forecast_run.Decoder.row``).
    :param value_cols, data_sha256: ``None`` for the synthetic bank; the column names and the file hash for a csv.
    :param time_col: ``None`` for the synthetic bank and for a csv without a time column; else the name of the one the run checked.
    :return: the record. It is NOT checked (:func:`check_record` does that).
    """
    return _record_of(
        FORECASTING,
        {
            "schema_version": SCHEMA_VERSION,
            "task": FORECASTING,
            "model": model,
            "family": family,
            "variant": variant,
            "lookback": int(lookback),
            "horizon": int(horizon),
            "channels": int(channels),
            "value_cols": None if value_cols is None else list(value_cols),
            "time_col": time_col,
            "scale_floor": float(scale_floor),
            "head": "point" if quantile_levels is None else "quantile",
            "quantile_levels": None if quantile_levels is None else [float(q) for q in quantile_levels],
            "point_index": point_index,
            "decode": decode,
            "data_format": data_format,
            "data_sha256": data_sha256,
        },
        variant_overrides,
    )


def build_depth_record(
    *,
    model: str,
    family: str,
    variant: str,
    input_size: Sequence[int],
    channels: int,
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a depth run; see the module contract.

    :param input_size: ``(height, width)`` the model was built at (the crops it trained on).
    :param channels: the model's input channels (3).
    :return: the record; ``normalize`` is :data:`DEPTH_NORMALIZE` and ``output`` the first of :data:`DEPTH_OUTPUTS`. It is
        NOT checked (:func:`check_record` does that).
    """
    return _record_of(
        DEPTH,
        {
            "schema_version": SCHEMA_VERSION,
            "task": DEPTH,
            "model": model,
            "family": family,
            "variant": variant,
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": DEPTH_NORMALIZE,
            "output": DEPTH_OUTPUTS[0],
        },
        variant_overrides,
    )


def build_pretrain_record(
    *,
    model: str,
    family: str,
    variant: str,
    input_size: Sequence[int],
    channels: int,
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: int,
    tokenizer: Optional[str],
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a pretrain run; see the module contract.

    :param input_size: ``(side, side)`` the model was built at.
    :param channels: the model's input channels (3).
    :param mean, std: the per-channel statistics the streams standardized with.
    :param patch_size: the side of a patch.
    :param tokenizer: the sha256 hex of the BEiT tokenizer file; ``None`` for a family with none.
    :return: the record; ``normalize`` is :data:`PRETRAIN_NORMALIZE` and ``output`` the first of
        :data:`PRETRAIN_OUTPUTS`. It is NOT checked (:func:`check_record` does that).
    """
    return _record_of(
        PRETRAIN,
        {
            "schema_version": SCHEMA_VERSION,
            "task": PRETRAIN,
            "model": model,
            "family": family,
            "variant": variant,
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": PRETRAIN_NORMALIZE,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "patch_size": int(patch_size),
            "tokenizer": tokenizer,
            "output": PRETRAIN_OUTPUTS[0],
        },
        variant_overrides,
    )


def build_tokenizer_record(
    *,
    model: str,
    family: str,
    variant: str,
    input_size: Sequence[int],
    channels: int,
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: int,
    num_embeddings: int,
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a ``beit_tokenizer`` run; see the module contract.

    :param input_size: ``(side, side)`` the model was built at.
    :param channels: the model's input channels (3).
    :param mean, std: the per-channel statistics the streams standardized with.
    :param patch_size: the side of the cell one code describes (the encoder's total stride).
    :param num_embeddings: the codebook size.
    :return: the record; ``normalize`` is :data:`PRETRAIN_NORMALIZE` and ``output`` the first of :data:`TOKENIZER_OUTPUTS`. It is NOT
        checked (:func:`check_record` does that).
    """
    return _record_of(
        BEIT_TOKENIZER,
        {
            "schema_version": SCHEMA_VERSION,
            "task": BEIT_TOKENIZER,
            "model": model,
            "family": family,
            "variant": variant,
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": PRETRAIN_NORMALIZE,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "patch_size": int(patch_size),
            "num_embeddings": int(num_embeddings),
            "output": TOKENIZER_OUTPUTS[0],
        },
        variant_overrides,
    )


def build_dino_record(
    *,
    model: str,
    family: str,
    variant: str,
    input_size: Sequence[int],
    channels: int,
    mean: Sequence[float],
    std: Sequence[float],
    patch_size: int,
    n_local_crops: int,
    dino_out_dim: int,
    embedding_dim: int,
    variant_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a ``dino`` run; see the module contract.

    :param input_size: ``(side, side)`` every view is rendered at (the global and the local crops alike).
    :param channels: the model's input channels (3).
    :param mean, std: the per-channel statistics the streams standardized with.
    :param patch_size: the side of a patch. :param n_local_crops: local crops per image, so the model takes ``2 + n_local_crops`` views.
    :param dino_out_dim: the projection head's width. :param embedding_dim: the backbone's CLS feature width.
    :return: the record; ``normalize`` is :data:`PRETRAIN_NORMALIZE` and ``output`` the first of :data:`DINO_OUTPUTS`. It is NOT checked
        (:func:`check_record` does that).
    """
    return _record_of(
        DINO,
        {
            "schema_version": SCHEMA_VERSION,
            "task": DINO,
            "model": model,
            "family": family,
            "variant": variant,
            "input_size": [int(side) for side in input_size],
            "channels": int(channels),
            "normalize": PRETRAIN_NORMALIZE,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "patch_size": int(patch_size),
            "n_local_crops": int(n_local_crops),
            "dino_out_dim": int(dino_out_dim),
            "embedding_dim": int(embedding_dim),
            "output": DINO_OUTPUTS[0],
        },
        variant_overrides,
    )


def _text_values(
    task: str, model: str, family: str, variant: str, seq_len: int, tokenizer: Mapping[str, Any], tokenizer_dir: Optional[str]
) -> Dict[str, Any]:
    """Return the values of :data:`_TEXT_KEYS` (and ``tokenizer_dir``) for a record of ``task``; ``tokenizer`` is a ``TokenizerSpec.record()``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "model": model,
        "family": family,
        "variant": variant,
        "seq_len": int(seq_len),
        "encoding": tokenizer["encoding"],
        "n_vocab": tokenizer["n_vocab"],
        "vocab_size": tokenizer["vocab_size"],
        "eot_id": tokenizer["eot_id"],
        "pad_id": tokenizer["pad_id"],
        "tokenizer": tokenizer["digest"],
        "tokenizer_files": dict(tokenizer["files"]),
        "tokenizer_dir": tokenizer_dir,
    }


def build_language_model_record(
    *,
    model: str,
    family: str,
    variant: str,
    seq_len: int,
    tokenizer: Mapping[str, Any],
    variant_overrides: Optional[Mapping[str, Any]] = None,
    tokenizer_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a language-model run; see the module contract.

    :param seq_len: the number of ids the model was built to take.
    :param tokenizer: :meth:`train.api.text_tokenizer.TokenizerSpec.record` of the run's tokenizer (its fields and digest).
    :param tokenizer_dir: the run's ``data.tokenizer_dir``; the record gets the ``tokenizer_dir`` key only when it is not ``None``.
    :return: the record; ``output`` is the first of :data:`LANGUAGE_OUTPUTS`. It is NOT checked (:func:`check_record` does that).
    """
    values = _text_values(LANGUAGE_MODEL, model, family, variant, seq_len, tokenizer, tokenizer_dir)
    return _record_of(LANGUAGE_MODEL, {**values, "output": LANGUAGE_OUTPUTS[0]}, variant_overrides)


def build_text_classification_record(
    *,
    model: str,
    family: str,
    variant: str,
    seq_len: int,
    tokenizer: Mapping[str, Any],
    class_names: Sequence[str],
    variant_overrides: Optional[Mapping[str, Any]] = None,
    tokenizer_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the ``task.json`` record of a text-classification run; see the module contract.

    :param seq_len: the number of ids a row holds, which the model was built to take.
    :param tokenizer: :meth:`train.api.text_tokenizer.TokenizerSpec.record` of the run's tokenizer (its ``cls_id`` is the id rows begin with).
    :param class_names: the label names in label order, one per output of the model.
    :param tokenizer_dir: as for :func:`build_language_model_record`.
    :return: the record; ``layout`` and ``output`` are the first of :data:`TEXT_LAYOUTS` and :data:`TEXT_CLASSIFICATION_OUTPUTS`. It is NOT
        checked (:func:`check_record` does that).
    """
    values = _text_values(TEXT_CLASSIFICATION, model, family, variant, seq_len, tokenizer, tokenizer_dir)
    extra = {"cls_id": tokenizer["cls_id"], "class_names": list(class_names), "layout": TEXT_LAYOUTS[0], "output": TEXT_CLASSIFICATION_OUTPUTS[0]}
    return _record_of(TEXT_CLASSIFICATION, {**values, **extra}, variant_overrides)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_json_plain(value: Any) -> bool:
    """Whether ``value`` is None, a bool, a finite number, a text, or a list or text-keyed dict of those."""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_plain(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_plain(item) for key, item in value.items())
    return False


def overrides_of_variant(table_row: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the keys of ``table_row`` that ``kwargs`` set to another value, as JSON-plain values.

    Interface contract: ``table_row`` is ``MODEL_VARIANTS[variant]`` and ``kwargs`` the factory
    keywords a run was built with (the resolver's output: the recipe's ``model_args``, the computed
    keys and the caller's ``model_args``). A key counts only when it is in the table AND its value
    differs from the table's (tuples read as lists), so a run that restates a table value, and
    every run that touches no table key, gets ``{}`` and therefore no ``variant_overrides`` key.
    Pure; imports nothing heavy.

    :param table_row: the variant's table.
    :param kwargs: the factory keywords.
    :return: ``{key: value}`` in the table's key order; ``{}`` when nothing was overridden.
    """
    return {
        key: _plain(kwargs[key])
        for key in table_row
        if key in kwargs and _plain(kwargs[key]) != _plain(table_row[key])
    }


def _is_whole(value: Any) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def older_record_reason(version: Any) -> Optional[str]:
    """Return the clause for a record of an older layout, or ``None`` when ``version`` is not one.

    Interface contract: ``version`` is a ``schema_version`` as parsed from JSON. An older layout is
    a whole number from 1 up to (not including) :data:`SCHEMA_VERSION`; anything else (a newer
    number, a bool, text, a missing value) is ``None`` and keeps the plain refusal. The clause is
    the one place the sentence is worded, so ``predict``, ``export`` and ``resume`` say the same.
    """
    if _is_whole(version) and 1 <= version < SCHEMA_VERSION:
        return (
            f"an older layout; this reader understands {SCHEMA_VERSION} only and cannot upgrade it, "
            "so the run has to be trained again"
        )
    return None


def _version_line(version: Any) -> str:
    older = older_record_reason(version)
    if older is not None:
        return f"'schema_version' is {version}, {older}"
    return f"'schema_version' is {version!r}; this reader understands {SCHEMA_VERSION} only"


def _header_problems(record: Mapping[str, Any], keys: Tuple[str, ...]) -> Tuple[List[str], bool]:
    """Return ``(lines, complete)``: the key-set problems, and whether no key is missing."""
    missing = [key for key in keys if key not in record]
    unknown = sorted(set(record) - set(keys) - set(OPTIONAL_RECORD_KEYS))
    lines: List[str] = []
    if missing:
        lines.append(f"missing key(s) {', '.join(map(repr, missing))}")
    if unknown:
        lines.append(f"unknown key(s) {', '.join(map(repr, unknown))}")
    return lines, not missing


def _identity_problems(record: Mapping[str, Any], keys: Tuple[str, ...]) -> List[str]:
    """Return the problems of ``task``, ``model``, ``family``, ``variant`` and, where ``keys`` hold it, ``class_names``."""
    lines: List[str] = []
    for key in ("task", "model", "family", "variant"):
        if not isinstance(record[key], str) or not record[key]:
            lines.append(f"{key!r} must be a non-empty text, got {record[key]!r}")
    if "class_names" in keys:
        names = record["class_names"]
        if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
            lines.append(f"'class_names' must be a non-empty list of texts, got {names!r}")
        elif len(set(names)) != len(names):
            repeated = sorted({n for n in names if names.count(n) > 1})
            lines.append(f"'class_names' repeats {', '.join(map(repr, repeated))}; each class needs its own name")
    if all(isinstance(record[key], str) and record[key] for key in ("model", "family", "variant")):
        if record["model"] != f"{record['family']}/{record['variant']}":
            lines.append(
                f"'model' is {record['model']!r} but 'family' and 'variant' give "
                f"{record['family'] + '/' + record['variant']!r}"
            )
    return lines


def _classification_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``input_shape`` and ``channels`` (``None`` when unusable)."""
    shape = record["input_shape"]
    shape_ok = (
        isinstance(shape, list)
        and len(shape) == 3
        and all(_is_whole(side) and side > 0 for side in shape)
        and shape[0] == shape[1]
    )
    if not shape_ok:
        return [f"'input_shape' must be [side, side, channels] of positive ints, got {shape!r}"], None
    if record["channels"] != shape[2]:
        return [f"'channels' is {record['channels']!r} but 'input_shape' has {shape[2]}"], shape[2]
    return [], shape[2]


def _input_size_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problem of ``input_size``, ``[height, width]`` of positive ints (a segmentation or a depth record)."""
    size = record["input_size"]
    if not (isinstance(size, list) and len(size) == 2 and all(_is_whole(side) and side > 0 for side in size)):
        return [f"'input_size' must be [height, width] of positive ints, got {size!r}"]
    return []


def _segmentation_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``input_size`` and ``channels`` (``None`` when unusable)."""
    lines = _input_size_problems(record)
    channels = record["channels"]
    if not (_is_whole(channels) and channels > 0):
        lines.append(f"'channels' must be a positive int, got {channels!r}")
        return lines, None
    return lines, channels


def _statistics_problems(record: Mapping[str, Any], width: Optional[int]) -> List[str]:
    """Return the problems of ``normalize``, ``mean`` and ``std`` against ``width`` channels."""
    lines: List[str] = []
    if record["normalize"] not in _NORMALIZE:
        lines.append(f"'normalize' must be one of {', '.join(_NORMALIZE)}, got {record['normalize']!r}")
    for key in ("mean", "std"):
        values = record[key]
        if not (isinstance(values, list) and values and all(_is_number(v) for v in values)):
            lines.append(f"{key!r} must be a list of finite numbers, got {values!r}")
        elif width is not None and len(values) != width:
            lines.append(f"{key!r} holds {len(values)} numbers for {width} channels")
    mean, std = record["mean"], record["std"]
    if isinstance(std, list) and all(_is_number(v) for v in std) and any(v <= 0 for v in std):
        lines.append(f"'std' must be positive, got {std!r}")
    for key, (low, high) in (("mean", MEAN_RANGE), ("std", STD_RANGE)):
        values = record[key]
        if not (isinstance(values, list) and all(_is_number(v) for v in values)):
            continue  # named above
        # A std of zero or less is already named above; do not say it twice.
        outside = [v for v in values if not low <= v <= high and (key == "mean" or v > 0)]
        if outside:
            lines.append(
                f"{key!r} must lie in [{low:g}, {high:g}] (the loader standardizes images in "
                f"[0, 1]), got {values!r}"
            )
    numbers = all(isinstance(v, list) and all(_is_number(x) for x in v) for v in (mean, std))
    if record["normalize"] == "unit" and numbers:
        if any(v != 0 for v in mean) or any(v != 1 for v in std):
            lines.append(
                f"'normalize' is 'unit' (no standardization) but 'mean' is {mean!r} and 'std' is "
                f"{std!r}; unit means zeros and ones"
            )
    return lines


def _overrides_problems(record: Mapping[str, Any]) -> List[str]:
    if "variant_overrides" not in record:
        return []
    overrides = record["variant_overrides"]
    if (
        isinstance(overrides, dict)
        and overrides
        and all(isinstance(key, str) and key and _is_json_plain(value) for key, value in overrides.items())
    ):
        return []
    return [
        "'variant_overrides' must be a non-empty object of text keys and JSON values "
        f"(it is written only when a model_args key changed a key of the variant's table), "
        f"got {_shown(overrides)}"
    ]


def _block_keys(name: str, block: Any, keys: Tuple[str, ...]) -> List[str]:
    """The line naming what ``block`` (an object at ``name``) misses of ``keys`` or holds beyond them."""
    if not isinstance(block, dict):
        return [f"{name!r} must be an object, got {_shown(block)}"]
    missing = [key for key in keys if key not in block]
    unknown = sorted(set(block) - set(keys))
    parts = ([f"missing key(s) {', '.join(map(repr, missing))}"] if missing else []) + (
        [f"unknown key(s) {', '.join(map(repr, unknown))}"] if unknown else [])
    return [f"{name!r} has {'; '.join(parts)}"] if parts else []


def _texts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _split_problems(split: Any) -> List[str]:
    """The problems of ``init.split``: counts for every relation but ``unknown``, which has none."""
    lines = _block_keys("init.split", split, INIT_SPLIT_KEYS)
    if lines:
        return lines
    if split["relation"] not in INIT_SPLIT_RELATIONS:
        lines.append(f"'init.split.relation' must be one of {', '.join(INIT_SPLIT_RELATIONS)}, "
                     f"got {_shown(split['relation'])}")
    counts = (split["n_select_overlap"], split["n_test_overlap"])
    if not all(n is None if split["relation"] == "unknown" else (type(n) is int and n >= 0)
               for n in counts):
        lines.append(f"'init.split' counts must be null for relation 'unknown' and counts otherwise, "
                     f"got {_shown(counts)}")
    if not (_texts([split["reason"]]) and (split["unit"] is None or _texts([split["unit"]]))):
        lines.append(f"'init.split.reason' must be a non-empty text and 'unit' a text or null, "
                     f"got {_shown(split['reason'])} and {_shown(split['unit'])}")
    if not (isinstance(split["keys"], dict) and _is_json_plain(split["keys"])):
        lines.append(f"'init.split.keys' must be an object of text keys and JSON values, got {_shown(split['keys'])}")
    return lines


def _init_problems(record: Mapping[str, Any]) -> List[str]:
    """The problems of the optional ``init`` block (keys in :data:`INIT_KEYS`); empty when the record has none."""
    if "init" not in record:
        return []
    block = record["init"]
    lines = _block_keys("init", block, INIT_KEYS)
    if lines:
        return lines
    for key in ("source", "file", "task", "family", "variant"):
        if not isinstance(block[key], str) or not block[key]:
            lines.append(f"'init.{key}' must be a non-empty text, got {_shown(block[key])}")
    if not (isinstance(block["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", block["sha256"])):
        lines.append(f"'init.sha256' must be 64 lowercase hex digits, got {_shown(block['sha256'])}")
    if block["relation"] not in INIT_RELATIONS:
        lines.append(f"'init.relation' must be one of {', '.join(INIT_RELATIONS)}, got {_shown(block['relation'])}")
    for key in ("geometry", "pinned"):
        if not (isinstance(block[key], dict) and _is_json_plain(block[key])):
            lines.append(f"'init.{key}' must be an object of text keys and JSON values, got {_shown(block[key])}")
    lines += _split_problems(block["split"])
    transfer = block["transfer"]
    keys = _block_keys("init.transfer", transfer, INIT_TRANSFER_KEYS)
    if keys:
        return lines + keys
    copied = transfer["copied"]
    if not isinstance(copied, int) or isinstance(copied, bool) or copied < 0:
        lines.append(f"'init.transfer.copied' must be a count, got {_shown(copied)}")
    lines += [f"'init.transfer.{key}' must be a list of weight paths, got {_shown(transfer[key])}"
              for key in ("reset", "grown", "dropped") if not _texts(transfer[key]) and transfer[key] != []]
    resampled = transfer["resampled"]
    if not isinstance(resampled, list) or not all(
            isinstance(row, dict) and set(row) == set(INIT_RESAMPLED_KEYS) and _texts([row["path"], row["rule"]])
            and all(isinstance(row[side], list) and all(is_count(n) for n in row[side]) for side in ("from", "to"))
            for row in resampled):
        lines.append(f"'init.transfer.resampled' must be a list of {{{', '.join(INIT_RESAMPLED_KEYS)}}} objects, "
                     f"got {_shown(resampled)}")
    return lines


def _eval_transform_problems(record: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    transform = record["eval_transform"]
    if not isinstance(transform, dict) or set(transform) != {"kind", "source_channels"}:
        lines.append(
            f"'eval_transform' must be an object with 'kind' and 'source_channels', got {transform!r}"
        )
        return lines
    if transform["kind"] not in EVAL_KINDS:
        lines.append(
            f"'eval_transform.kind' is {transform['kind']!r}, which this predictor does not "
            f"reproduce (it reproduces {', '.join(EVAL_KINDS)})"
        )
    if transform["source_channels"] not in (1, 3) or isinstance(transform["source_channels"], bool):
        lines.append(
            f"'eval_transform.source_channels' must be 1 or 3, got {transform['source_channels']!r}"
        )
    return lines


def _segmentation_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``ignore_index``, ``mask_threshold`` and ``output``."""
    lines: List[str] = []
    names = record["class_names"]
    classes = len(names) if isinstance(names, list) and names else None
    ignore = record["ignore_index"]
    if ignore is not None and not (_is_whole(ignore) and ignore >= 0):
        lines.append(f"'ignore_index' must be null or a whole number of 0 or more, got {ignore!r}")
    elif ignore is not None and classes is not None and ignore < classes:
        lines.append(
            f"'ignore_index' is {ignore}, a class index; it must lie outside the class indices "
            f"0..{classes - 1}"
        )
    threshold = record["mask_threshold"]
    if threshold is not None:
        if not _is_number(threshold):
            lines.append(f"'mask_threshold' must be null or a finite number, got {threshold!r}")
        else:
            if classes is not None and classes != 2:
                lines.append(f"'mask_threshold' needs exactly 2 classes, the record names {classes}")
            if ignore is not None:
                lines.append(
                    "'mask_threshold' and 'ignore_index' cannot both be set: a thresholded mask "
                    "has two classes and no void"
                )
    if record["output"] not in SEGMENTATION_OUTPUTS:
        lines.append(
            f"'output' must be one of {', '.join(SEGMENTATION_OUTPUTS)}, got {record['output']!r}"
        )
    return lines


def _denoising_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``channels`` and ``size_multiple`` (``None`` when ``channels`` is unusable)."""
    from .denoising_args import CHANNEL_COUNTS

    lines: List[str] = []
    channels = record["channels"]
    usable = _is_whole(channels) and channels in CHANNEL_COUNTS
    if not usable:
        lines.append(f"'channels' must be one of {', '.join(map(str, CHANNEL_COUNTS))}, got {channels!r}")
    multiple = record["size_multiple"]
    if not (_is_whole(multiple) and multiple > 0):
        lines.append(f"'size_multiple' must be a positive int, got {multiple!r}")
    return lines, channels if usable else None


def _denoising_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``data_range`` and ``output``."""
    lines: List[str] = []
    stamp = unit_domain_stamp()
    if record["data_range"] != stamp:
        lines.append(
            f"'data_range' must be {stamp!r} (the only pixel domain a denoiser of this task is trained "
            f"on), got {record['data_range']!r}"
        )
    if record["output"] not in DENOISING_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(DENOISING_OUTPUTS)}, got {record['output']!r}")
    return lines


def _forecast_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for the window, the channel count and the scale floor."""
    lines = [
        f"{key!r} must be a positive int, got {record[key]!r}"
        for key in ("lookback", "horizon", "channels")
        if not (_is_whole(record[key]) and record[key] > 0)
    ]
    floor = record["scale_floor"]
    if not (_is_number(floor) and floor >= 0):
        lines.append(f"'scale_floor' must be a finite number of at least 0, got {floor!r}")
    channels = record["channels"]
    return lines, channels if _is_whole(channels) and channels > 0 else None


def _forecast_head_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``head``, ``quantile_levels`` and ``point_index``: a quantile head has both, a point head neither."""
    head, levels, index = record["head"], record["quantile_levels"], record["point_index"]
    if head not in FORECAST_HEADS:
        return [f"'head' must be one of {', '.join(FORECAST_HEADS)}, got {head!r}"]
    if head == "point":
        return [f"{key!r} must be null for a point head, got {value!r}"
                for key, value in (("quantile_levels", levels), ("point_index", index)) if value is not None]
    if not isinstance(levels, list) or not levels:
        return [f"'quantile_levels' must be a list for a quantile head, got {levels!r}"]
    if not all(_is_number(q) and 0 < q < 1 for q in levels):
        return [f"'quantile_levels' must be a list of numbers between 0 and 1, got {levels!r}"]
    if not (_is_whole(index) and 0 <= index < len(levels)):
        return [f"'point_index' must be a whole number below the {len(levels)} quantile levels, got {index!r}"]
    return []


def _forecast_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of the head, the decode row and where the data came from."""
    lines = _forecast_head_problems(record)
    if not isinstance(record["decode"], str) or not record["decode"]:
        lines.append(f"'decode' must be a non-empty text, got {record['decode']!r}")
    kind, columns, digest = record["data_format"], record["value_cols"], record["data_sha256"]
    time_col = record["time_col"]
    if kind not in FORECAST_FORMATS:
        return lines + [f"'data_format' must be one of {', '.join(FORECAST_FORMATS)}, got {kind!r}"]
    if kind == FORECAST_FORMATS[0]:
        return lines + [f"{key!r} must be null for the {kind} format, got {value!r}"
                        for key, value in (("value_cols", columns), ("time_col", time_col), ("data_sha256", digest))
                        if value is not None]
    if time_col is not None and not (isinstance(time_col, str) and time_col):
        lines.append(f"'time_col' must be a non-empty text or null for the {kind} format, got {time_col!r}")
    width = record["channels"]
    if not (isinstance(columns, list) and len(columns) == width and len(set(columns)) == width
            and all(isinstance(c, str) and c for c in columns)):
        lines.append(f"'value_cols' must be a list of {width} distinct texts for the {kind} format, got {columns!r}")
    if not (isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")):
        lines.append(f"'data_sha256' must be the 64 hex digits of the csv's sha256, got {digest!r}")
    return lines


def _depth_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for ``input_size`` and ``channels``, which must be :data:`DEPTH_CHANNELS`."""
    lines = _input_size_problems(record)
    channels = record["channels"]
    if not (_is_whole(channels) and channels == DEPTH_CHANNELS):
        lines.append(f"'channels' must be {DEPTH_CHANNELS} (the model takes RGB), got {channels!r}")
        return lines, None
    return lines, channels


def _depth_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``normalize`` and ``output``."""
    lines: List[str] = []
    if record["normalize"] != DEPTH_NORMALIZE:
        lines.append(
            f"'normalize' must be {DEPTH_NORMALIZE!r} (the ImageNet mean and standard deviation of the DINOv2 input, the "
            f"only standardization a depth run trains with), got {record['normalize']!r}"
        )
    if record["output"] not in DEPTH_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(DEPTH_OUTPUTS)}, got {record['output']!r}")
    return lines


def _pretrain_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, channels)`` for a square ``input_size``, a ``patch_size`` that divides it and 3 ``channels``."""
    lines = _input_size_problems(record)
    size, patch, channels = record["input_size"], record["patch_size"], record["channels"]
    if not lines and size[0] != size[1]:
        lines.append(f"'input_size' must be [side, side] (the pretrain streams make square images), got {size!r}")
    if not (_is_whole(patch) and patch > 0):
        lines.append(f"'patch_size' must be a positive int, got {patch!r}")
    elif not lines and size[0] % patch:
        lines.append(
            f"'patch_size' {patch} does not divide 'input_size' {size}: the model cuts the image into whole patches")
    if not (_is_whole(channels) and channels == PRETRAIN_CHANNELS):
        lines.append(f"'channels' must be {PRETRAIN_CHANNELS} (the pretrain streams are RGB), got {channels!r}")
        return lines, None
    return lines, channels


def _streams_normalize_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problem of ``normalize`` when it is a known word other than :data:`PRETRAIN_NORMALIZE`: the pretrain streams, which a
    ``pretrain`` run and a ``beit_tokenizer`` run both read, always standardize."""
    if record["normalize"] in _NORMALIZE and record["normalize"] != PRETRAIN_NORMALIZE:
        return [
            f"'normalize' must be {PRETRAIN_NORMALIZE!r} (the pretrain streams standardize with the dataset's mean and "
            f"std), got {record['normalize']!r}"
        ]
    return []


def _pretrain_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``normalize`` (a known word other than :data:`PRETRAIN_NORMALIZE`), ``tokenizer`` and
    ``output``."""
    lines = _streams_normalize_problems(record)
    tokenizer = record["tokenizer"]
    if tokenizer is not None and not (isinstance(tokenizer, str) and re.fullmatch(r"[0-9a-f]{64}", tokenizer)):
        lines.append(
            f"'tokenizer' must be null or the 64 lowercase hex digits of the tokenizer file's sha256, got {tokenizer!r}"
        )
    if record["output"] not in PRETRAIN_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(PRETRAIN_OUTPUTS)}, got {record['output']!r}")
    return lines


def _tokenizer_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``normalize`` (a known word other than :data:`PRETRAIN_NORMALIZE`), ``num_embeddings`` and ``output``."""
    lines = _streams_normalize_problems(record)
    codes = record["num_embeddings"]
    if not (_is_whole(codes) and codes > 0):
        lines.append(f"'num_embeddings' must be a positive int (the codebook size), got {codes!r}")
    if record["output"] not in TOKENIZER_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(TOKENIZER_OUTPUTS)}, got {record['output']!r}")
    return lines


def _dino_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``normalize`` (a known word other than :data:`PRETRAIN_NORMALIZE`), the three counts and ``output``."""
    lines = _streams_normalize_problems(record)
    for key in ("n_local_crops", "dino_out_dim", "embedding_dim"):
        value = record[key]
        if not (_is_whole(value) and value > 0):
            lines.append(f"{key!r} must be a positive int, got {value!r}")
    if record["output"] not in DINO_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(DINO_OUTPUTS)}, got {record['output']!r}")
    return lines


def _text_geometry(record: Mapping[str, Any], id_keys: Sequence[str]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, None)`` for ``seq_len``, the vocabulary and the ``id_keys`` ids in it (a text record has no channels)."""
    lines = [
        f"{key!r} must be a positive int, got {record[key]!r}"
        for key in ("seq_len", "n_vocab", "vocab_size")
        if not (_is_whole(record[key]) and record[key] > 0)
    ]
    if lines:
        return lines, None
    vocab = record["vocab_size"]
    if record["n_vocab"] > vocab:
        lines.append(f"'n_vocab' {record['n_vocab']} is above 'vocab_size' {vocab}: the model's vocabulary holds every id of the tokenizer")
    lines += [
        f"{key!r} must be a whole number below 'vocab_size' {vocab}, got {record[key]!r}"
        for key in id_keys
        if not (_is_whole(record[key]) and 0 <= record[key] < vocab)
    ]
    return lines, None


def _language_model_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, None)`` for ``seq_len``, the vocabulary and the ids in it (a language record has no channels)."""
    return _text_geometry(record, ("eot_id", "pad_id"))


def _text_classification_geometry(record: Mapping[str, Any]) -> Tuple[List[str], Optional[int]]:
    """Return ``(lines, None)`` for the language model's keys and ``cls_id``, which must not be the pad id (the mask would hide it)."""
    lines, _ = _text_geometry(record, ("eot_id", "pad_id", "cls_id"))
    if not lines and record["cls_id"] == record["pad_id"]:
        lines.append(f"'cls_id' and 'pad_id' are both {record['pad_id']}: the padding mask would hide the [CLS] every row begins with")
    return lines, None


def _tokenizer_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of ``encoding``, ``tokenizer`` and ``tokenizer_files`` of a text record: formats, not the code's pin."""
    # DECISION plan-2026-11-24T092348-7806a290/D-033
    # The FORMAT of the tokenizer keys is checked here and nothing about what this code pins for the encoding: a record of another
    # tokenizer (an ascii95 run beside a gpt2 one, a digest this code no longer pins) is a valid record that ``export`` and the
    # ``init`` identity read, and refusing it here would hide the sentence that names both digests. ``predict`` checks the digest
    # against the pin when it opens the tokenizer. Do NOT refuse a record because its digest is not the pinned one, and do NOT copy
    # this check into the task's tail check: the language model and the text classifier both call it (D-062).
    lines: List[str] = []
    if record["encoding"] not in TOKENIZERS:
        lines.append(f"'encoding' must be one of {', '.join(TOKENIZERS)}, got {record['encoding']!r}")
    digest, files = record["tokenizer"], record["tokenizer_files"]
    if not (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)):
        lines.append(f"'tokenizer' must be the 64 lowercase hex digits of the tokenizer's digest, got {digest!r}")
    if not (isinstance(files, dict) and all(
            isinstance(name, str) and name and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
            for name, sha in files.items())):
        lines.append(f"'tokenizer_files' must be an object of cache file name to the 64 lowercase hex digits of its sha256, got {_shown(files)}")
    elif record["encoding"] in TOKENIZERS and bool(files) != (record["encoding"] != ASCII95):
        lines.append(f"'tokenizer_files' must be {'empty' if record['encoding'] == ASCII95 else 'the BPE files'} for the {record['encoding']} "
                     f"tokenizer, got {_shown(files)}")
    if "tokenizer_dir" in record and not (isinstance(record["tokenizer_dir"], str) and Path(record["tokenizer_dir"]).is_absolute()):
        lines.append(f"'tokenizer_dir' must be an absolute directory name (written only when data.tokenizer_dir was set), got {_shown(record['tokenizer_dir'])}")
    # DECISION plan-2026-11-24T092348-7806a290/D-078
    # A record that names the digest this code PINS for its encoding holds the ids that digest is made of: `eot_id`, `pad_id` and `cls_id`
    # are what `predict` appends and pads with and what a consumer of an exported record builds rows from, so another id is a forged
    # or corrupt record, not another tokenizer. It holds the digest's files too (D-085: `predict` opens a directory whose files hash to
    # the record's `tokenizer_files`, which are therefore the pin's). A record of another digest is judged by format alone (D-033). Do NOT
    # compare the ids or the files of a record whose digest differs (it is a valid record of a tokenizer this code does not pin).
    spec = tokenizer_spec(record["encoding"]) if record["encoding"] in TOKENIZERS else None
    if spec is not None and digest == spec.digest:
        lines += [f"{key!r} is {record[key]!r}, but the tokenizer digest the record names is {spec.encoding}'s, whose {key} is {getattr(spec, key)}"
                  for key in ("eot_id", "pad_id", "cls_id") if key in record and record[key] != getattr(spec, key)]
        if isinstance(files, dict) and files != dict(spec.files):
            lines.append(f"'tokenizer_files' is {_shown(files)}, but the tokenizer digest the record names is {spec.encoding}'s, whose files are {dict(spec.files)}")
    return lines


def _language_model_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of the tokenizer keys and ``output`` of a language-model record."""
    lines = _tokenizer_problems(record)
    if record["output"] not in LANGUAGE_OUTPUTS:
        lines.append(f"'output' must be one of {', '.join(LANGUAGE_OUTPUTS)}, got {record['output']!r}")
    return lines


def _text_classification_tail_problems(record: Mapping[str, Any]) -> List[str]:
    """Return the problems of the tokenizer keys, ``layout`` and ``output`` of a text-classification record."""
    lines = _tokenizer_problems(record)
    for key, allowed in (("layout", TEXT_LAYOUTS), ("output", TEXT_CLASSIFICATION_OUTPUTS)):
        if record[key] not in allowed:
            lines.append(f"{key!r} must be one of {', '.join(allowed)}, got {record[key]!r}")
    return lines


# DECISION plan-2026-11-10T095307-d483ffab/D-006: the per-task work of reading a record is a ROW of
# a table keyed by task, not a branch. Do NOT add a third ``elif task == ...`` here or in
# ``input_shape_of``, ``load_model`` or ``_run_on_zeros`` (a third task would be one more branch in
# five places), and do NOT run the identity (``class_names``) and statistics (``normalize``,
# ``mean``, ``std``) checks for every task: they run only where the task's ``RECORD_KEYS`` hold
# those keys, so a task whose record has no class count is checked by its own rows alone. A task
# missing from any table in ``TASK_TABLES`` is refused naming the tasks that table knows.
_STATISTICS_KEYS = ("normalize", "mean", "std")


def known_tasks_line(task: Any, table: Mapping[str, Any], subject: str = "this reader") -> str:
    """Return the sentence that refuses ``task`` and names the tasks ``table`` knows (its keys)."""
    return f"'task' is {task!r}; {subject} knows {', '.join(table)}"


def record_problems(record: Mapping[str, Any]) -> List[str]:
    """Return one line per way ``record`` is not a valid version-2 record (empty when valid)."""
    version = record.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        return [_version_line(version)]
    task = record.get("task")
    for table in TASK_TABLES.values():
        if not isinstance(task, str) or task not in table:
            return [known_tasks_line(task, table)]
    keys = RECORD_KEYS[task]
    lines, complete = _header_problems(record, keys)
    if not complete:
        return lines
    lines += _identity_problems(record, keys)
    geometry, width = GEOMETRY_CHECKS[task](record)
    lines += geometry
    if all(key in keys for key in _STATISTICS_KEYS):
        lines += _statistics_problems(record, width)
    lines += _overrides_problems(record)
    lines += _init_problems(record)
    lines += TAIL_CHECKS[task](record)
    return lines


def check_record(record: Any, source: Union[str, Path]) -> None:
    """Raise :class:`ConfigError` unless ``record`` is a valid version-2 record; see the module contract."""
    lines = ["not a JSON object"] if not isinstance(record, dict) else record_problems(record)
    if lines:
        raise ConfigError(f"Invalid {TASK_FILE} {source}: " + "; ".join(lines))


def refuse_old_record(path: Path, prefix: str) -> None:
    """Raise :class:`ConfigError` when ``path`` is a record of an older layout; see the module contract."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return
    older = older_record_reason(record.get("schema_version")) if isinstance(record, dict) else None
    if older is not None:
        raise ConfigError(
            f"{prefix}: {path} has 'schema_version' {record['schema_version']}, {older}"
        )


def read_record(path: Path) -> Dict[str, Any]:
    """Return the checked record in the file ``path``.

    :raises ConfigError: when the file is missing, is not JSON, or fails :func:`check_record`.
    """
    if not path.is_file():
        raise ConfigError(
            f"Invalid value for 'model': {path} does not exist; a model is loaded from a run "
            f"directory, or from a .keras file whose directory holds {TASK_FILE}"
        )
    try:
        record = json.loads(path.read_text())
    except ValueError as exc:
        raise ConfigError(f"Invalid {TASK_FILE} {path}: not JSON ({exc})") from exc
    check_record(record, path)
    return record


#: The data keys the two text tasks share: the length, the encoding, the vocabulary and the tokenizer digest.
_TEXT_DATA_KEYS = ("seq_len", "encoding", "vocab_size", "tokenizer")

#: The keys of a record that describe the DATA a run was trained on, per task (a resume compares
#: them). A segmentation run adds the void and the threshold: a different ``ignore_index`` is
#: other labels; a denoising run holds the channel count its files are decoded to and the pixel domain.
#: ``output`` is how the predictor answers, not what the run saw.
# DECISION plan-2026-11-10T095307-d483ffab/D-027
# A denoising resume compares ``channels`` and ``data_range`` and NOTHING about the held-out sets
# (``data.test_dirs``). Do NOT add a ``test_dirs`` or a held-out fingerprint key: the sets weigh on no
# weight (they are listed and scored once, after ``fit``, best and final model together, in whichever
# process finishes the run), the request comparison already refuses another ``test_dirs`` value, and the
# files of train and validation are covered by ``data_counts`` and ``data_fingerprint``. A held-out file
# added between a kill and a resume changes the ``n`` the cells report, and the report states it.
DATA_KEYS: Dict[str, Tuple[str, ...]] = {
    CLASSIFICATION: ("class_names", "input_shape", "normalize", "eval_transform", "mean", "std"),
    SEGMENTATION: (
        "class_names", "input_size", "normalize", "mean", "std", "ignore_index", "mask_threshold",
    ),
    DENOISING: ("channels", "data_range"),
    # DECISION plan-2026-11-13T105600-3a8319ef/D-010
    # A forecasting resume compares the window, the channels and the columns, the scale floor and where the data came from
    # (the csv's bytes), NOT the head, the levels or the decode row: those are the model's, and the resume builds the model
    # from the request, which the request comparison has already held equal. Do NOT add them "for completeness".
    FORECASTING: ("lookback", "horizon", "channels", "value_cols", "scale_floor", "data_format", "data_sha256"),
    # A depth resume compares the size the model is built at, the channels and the standardization: what the crops were.
    DEPTH: ("input_size", "channels", "normalize"),
    # A pretrain resume compares the geometry, the scaling, the patch and the tokenizer: the masks and the targets are
    # functions of them. ``output`` is how the predictor answers.
    PRETRAIN: ("input_size", "channels", "normalize", "mean", "std", "patch_size", "tokenizer"),
    # A language-model resume compares the length, the encoding, the vocabulary and the tokenizer digest (which holds the ids and the
    # file hashes): the windows are a function of them. ``output`` is how the predictor answers.
    LANGUAGE_MODEL: _TEXT_DATA_KEYS,
    # A text-classification resume compares those and the class names (the labels): the rows are a function of them. ``cls_id`` is held
    # by the digest, ``layout`` and ``output`` are how the rows are made and the model answers.
    TEXT_CLASSIFICATION: (*_TEXT_DATA_KEYS, "class_names"),
    # A tokenizer resume compares the geometry, the scaling and the patch: the images and the cells a code describes are functions of them.
    # NOT the codebook (the model's: the request comparison has held it) and not ``output`` (how the predictor answers) (D-044).
    BEIT_TOKENIZER: ("input_size", "channels", "normalize", "mean", "std", "patch_size"),
    # A dino resume compares the geometry, the scaling, the patch and the number of local crops: the views are functions of them. NOT the head's
    # width or the embedding's (the model's: the request comparison has held them) and not ``output``.
    DINO: ("input_size", "channels", "normalize", "mean", "std", "patch_size", "n_local_crops"),
}


def _shown(value: Any) -> str:
    text = json.dumps(value, default=repr)
    return text if len(text) <= 100 else text[:97] + "..."


def record_differences(earlier: Mapping[str, Any], now: Mapping[str, Any]) -> List[str]:
    """List the ``DATA_KEYS`` where ``now`` differs from ``earlier``; see the module contract."""
    if earlier["task"] != now["task"]:
        return [f"task: {_shown(earlier['task'])} -> {_shown(now['task'])}"]
    lines = []
    for key in DATA_KEYS[now["task"]]:
        before, after = earlier[key], now[key]
        if key in ("mean", "std"):
            same = len(before) == len(after) and all(
                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-7) for a, b in zip(before, after)
            )
        else:
            same = before == after
        if not same:
            lines.append(f"{key}: {_shown(before)} -> {_shown(after)}")
    return lines


def _task_candidates(model_file: Path) -> Tuple[Path, ...]:
    """Return the records a ``.keras`` file may be read with, best first (see the module contract)."""
    # DECISION plan-2026-10-30T132432-a865fa08/D-068: ``<file>.task.json`` first, because that is
    # the name ``export`` writes for ``<out>``; ``<stem>.task.json`` and ``task.json`` after it.
    # Do NOT read only ``task.json``: an exported file moved away from its run has none.
    return (
        model_file.with_name(model_file.name + TASK_SUFFIX),
        model_file.with_name(model_file.stem + TASK_SUFFIX),
        model_file.parent / TASK_FILE,
    )


def _model_files(path: Path) -> Tuple[Path, Path]:
    """Return ``(model file, task file)`` for a run directory or a ``.keras`` file."""
    from train.common.callbacks import best_checkpoint_path

    if path.is_dir():
        model_file = Path(best_checkpoint_path(str(path)))
        if not model_file.is_file():
            raise ConfigError(
                f"Invalid value for 'model': {path} holds no {model_file.name} (the run had "
                "checkpoint_args.save_best off?); pass the .keras file you want, e.g. "
                f"{path / 'final_model.keras'}"
            )
        return model_file, path / TASK_FILE
    if path.is_file():
        candidates = _task_candidates(path)
        for candidate in candidates:
            if candidate.is_file():
                return path, candidate
        raise ConfigError(
            f"Invalid value for 'model': {path} has no record beside it; a .keras file is read "
            f"with the first of {', '.join(c.name for c in candidates)} that exists in "
            f"{path.parent} (export writes {candidates[0].name})"
        )
    raise ConfigError(f"Invalid value for 'model': {path} is neither a run directory nor a file")


def record_beside(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the checked record of a run directory or ``.keras`` file; see the module contract."""
    return read_record(_model_files(Path(path))[1])


def _classification_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    return tuple(record["input_shape"])


def _segmentation_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    return (*record["input_size"], record["channels"])


def _forecast_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    return (record["lookback"], record["channels"])


def _dino_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    """The rank-5 input of the TRAINING model a ``dino`` run saves: ``(2 + n_local_crops, height, width, channels)`` (a view set)."""
    return (2 + record["n_local_crops"], *record["input_size"], record["channels"])


def input_shape_of(record: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return ``(height, width, channels)``, the model input a checked record names."""
    return INPUT_SHAPE_OF[record["task"]](record)


def load_model(path: Union[str, Path]) -> "Predictor":
    """Load a trained model and its ``task.json``; see the module contract."""
    import keras

    from . import recipes

    model_file, task_file = _model_files(Path(path))
    record = read_record(task_file)
    # The family's model code is imported BEFORE the Keras file is read, so the classes
    # the archive names are registered whatever the caller imported (D-051).
    recipes.register_builtin_models()
    spec = registry.MODELS.get(record["family"])
    if spec is None:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: the family {record['family']!r} is not registered "
            f"(registered: {', '.join(sorted(registry.MODELS)) or 'none'})"
        )
    if record["task"] not in spec.tasks:
        # A record whose family trains another task is not one a run wrote (the trainer refuses the
        # pair before it claims the directory, D-072); say so here rather than let the family check
        # below compare a model with a variants table that is not its own.
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: the family {record['family']!r} trains "
            f"{' and '.join(spec.tasks)}, not {record['task']!r}; the file is not from one run"
        )
    spec.load_factory()
    # DECISION plan-2026-10-30T132432-a865fa08/D-061: ``compile=False``. Do NOT load the compile config: a
    # predictor needs no loss or metric, and a metric class the archive names (the
    # library's half-precision top-k, D-051) would be one more thing that can fail a
    # reload that needs only the weights.
    check_model_file(model_file, "Invalid value for 'model'")
    model = keras.models.load_model(str(model_file), compile=False)
    check_model_against_record(model, record, spec, model_file, task_file)
    MODEL_CHECKS[record["task"]](model, record, model_file, task_file)
    output = _run_on_zeros(model, record, model_file, task_file)
    OUTPUT_CHECKS[record["task"]](output, record)
    return PREDICTORS[record["task"]](model, record)


def _plain(value: Any) -> Any:
    """Return ``value`` with every tuple a list, so a config and a variant table compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def check_model_against_record(
    model: Any, record: Mapping[str, Any], spec: "registry.ModelSpec", model_file: Path, task_file: Path
) -> None:
    """Raise :class:`ConfigError` unless ``model`` is the family and variant ``record`` names.

    Interface contract: ``model`` is a loaded Keras model, ``record`` a checked record, ``spec`` the
    registered :class:`~train.api.registry.ModelSpec` of ``record["family"]``. The family is the
    class named by ``spec.variants_attr`` (``model`` must be an instance of it); the variant must
    be a key of that class's ``MODEL_VARIANTS``. The EXPECTED table is the variant's table updated
    by the record's ``variant_overrides`` (each override key must be a key of the table, and the
    model's ``get_config()`` must report it), and every key of the expected table that the model's
    ``get_config()`` also holds must be equal. Measured on all six families (the smallest variant of
    each, against every variant's table): a wrong variant differs in at least one such key.
    A run trained with ``model_args`` that override table keys is honoured through
    ``variant_overrides`` (D-032, which replaces the refusal D-028 accepted for it); a model
    that differs from its variant in a key the record does not list is refused. Two variants that
    differ only in keys the record overrides are the same architecture and cannot be told apart.

    A family whose models are functional (the four segmentation families: no class to be an instance of,
    no table keys in ``get_config()``) is checked by what its layers show: ``spec.observe(model)`` names
    the table keys the layers carry, or ``None`` for a model that holds no layer of the family, and
    each key it returns must equal the expected table's. A family with an ``observe`` is checked this way
    even when its ``variants_attr`` names a class (``resnet_seg`` reads ``ResNet.MODEL_VARIANTS``; its model
    wraps a ``ResNet``, it is not one). The keys a layer does not show
    (ConvUNext's ``drop_path_rate``) and a model that differs in one the record does not list are not
    seen; two variants that differ only there are not told apart (D-088).
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-028
    # The family and the variant of the record are compared with the MODEL, not only with each
    # other (``record_problems`` already checks ``model == family/variant``): a record that names
    # another family or variant still predicts, with a lie in its metadata. Do NOT compare the
    # whole config: ``model_args`` that are not part of a variant table are free.
    # DECISION plan-2026-10-31T180033-2faf890a/D-032
    # Supersedes D-028's accepted refusal of a model whose table keys ``model_args`` overrode.
    # Do NOT refuse such a model: the run records the overridden keys in ``variant_overrides``
    # and the EXPECTED table is the variant's updated by them. Refusing it made a run that
    # trained for hours unable to ``predict`` or ``export`` (step 14.7, measured on 5 of 6
    # families). A wrong variant, a foreign key and an override the model lacks stay refused.
    from .registry import load_attr

    module_name, _, attr_path = spec.variants_attr.partition(":")
    family, variant = record["family"], record["variant"]
    # DECISION plan-2026-10-31T180033-2faf890a/D-080
    # A family whose table is a module-level mapping (``module:TABLE``, no class: the functional
    # U-Nets of the segmentation task) has no class to be an instance of and no table keys in its
    # ``get_config()`` (a functional model reports its layers). Do NOT rebuild the variant and
    # compare weight shapes (a ``model_args`` key that is not a table key, ``kernel_size`` say,
    # changes the shapes and the record does not carry it: D-032's failure, a run trained for hours
    # that cannot ``predict``), and do NOT compare ``model.name`` (``model_args.model_name`` renames
    # the model).
    # DECISION plan-2026-10-31T180033-2faf890a/D-088
    # D-080 stopped there and README.md said the family "cannot be checked". It can: the two
    # families share no layer class (ConvUNext: ConvStem, DownsampleAndSkip, ConvNextV2Block; AccUNet:
    # AccUNet), so `spec.observe(model)` reads the family and the table keys the layers show, and a
    # mismatch is refused. Do NOT go back to "the variant is in the table and the model takes the
    # size": a convunext/tiny file loaded under records naming convunext/base, convunext/xlarge and
    # accunet/base, and `export` wrote the wrong record beside the export (review-iter-3 concern 3).
    # DECISION plan-2026-11-03T221936-98b866fb/D-012
    # A family with `observe` is checked by what its layers show EVEN WHEN its table lives on a class
    # (`resnet_seg` and `convnext_seg` share the classifier's `ResNet.MODEL_VARIANTS` and
    # `ConvNeXtV1.MODEL_VARIANTS`: one table, not a copy). Their model is a functional wrapper, so it is not
    # an instance of that class and `isinstance` would refuse every honest file. Do NOT copy the table into
    # `segmentation_models` to make `owner` None, and do NOT drop the `isinstance` check for the classifiers.
    owner = load_attr(f"{module_name}:{attr_path.rsplit('.', 1)[0]}") if "." in attr_path else None
    observed = spec.observe is not None
    if owner is not None and not observed and not isinstance(model, owner):
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the family {family!r} ({owner.__name__}) "
            f"but {model_file} holds a {type(model).__name__}; the file and the model are not "
            "from one run"
        )
    table = spec.load_variants()
    if variant not in table:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: {variant!r} is not a variant of {family!r} "
            f"(known: {', '.join(sorted(table))})"
        )
    # DECISION plan-2026-11-13T105600-3a8319ef/D-010
    # A variants table may be a tuple of NAMES (the ets family: ``ETS_VARIANTS``), which has no row to index and no keys to
    # compare: ``wanted`` is then ``None`` and only the name is checked. Do NOT index it by name (a TypeError on every ets
    # file) and do NOT copy the tuple into a mapping to make it fit.
    wanted = table[variant] if isinstance(table, Mapping) else None
    overrides = record.get("variant_overrides") or {}
    if overrides and not isinstance(wanted, Mapping):
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it records variant_overrides but {variant!r} of "
            f"{family!r} has no table of keys to override"
        )
    if isinstance(wanted, Mapping):
        foreign = [key for key in overrides if key not in wanted]
        if foreign:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: variant_overrides names {', '.join(map(repr, foreign))}, "
                f"which {variant!r} of {family!r} does not have (it has {', '.join(map(repr, wanted))})"
            )
        if owner is None or observed:
            _check_observed(model, spec, {**wanted, **overrides}, overrides, record, model_file, task_file)
            return
        built = model.get_config()
        unreported = [key for key in overrides if key not in built]
        if unreported:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: variant_overrides names {', '.join(map(repr, unreported))} "
                f"but {model_file} does not report that key, so the override cannot be true of it"
            )
        expected = {**wanted, **overrides}
        differing = [
            f"{key} is {_shown(_plain(built[key]))} in the model, {_shown(_plain(value))} in "
            f"{variant!r}{' with the recorded overrides' if overrides else ''}"
            for key, value in expected.items()
            if key in built and _plain(built[key]) != _plain(value)
        ]
        if differing:
            raise ConfigError(
                f"Invalid {TASK_FILE} {task_file}: it names the variant {variant!r} but "
                f"{model_file} is not that variant ({'; '.join(differing)}); the file and the "
                "model are not from one run"
            )


def _check_observed(
    model: Any, spec: "registry.ModelSpec", expected: Mapping[str, Any], overrides: Mapping[str, Any],
    record: Mapping[str, Any], model_file: Path, task_file: Path,
) -> None:
    """Refuse a functional ``model`` whose layers are not the family and variant of ``record``.

    Interface contract: ``expected`` is the variant's table updated by the record's overrides,
    ``spec.observe`` the family's reader (``None``: nothing can be read, no check). The reader's keys
    that are also in ``expected`` must be equal. Raises :class:`ConfigError` naming the file, the
    family or the differing keys; the file and the model are "not from one run".
    """
    observer = spec.load_observer()
    if observer is None:
        return
    family, variant = record["family"], record["variant"]
    seen = observer(model)
    if seen is None:
        own = sorted({
            type(layer).__name__ for layer in getattr(model, "layers", None) or []
            if not type(layer).__module__.startswith(("keras", "tensorflow"))
        })
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the family {family!r} but {model_file} holds "
            f"no layer of that family (its own layers: {', '.join(own[:8]) or 'none'}); the file and "
            "the model are not from one run"
        )
    differing = [
        f"{key} is {_shown(_plain(value))} in the model, {_shown(_plain(expected[key]))} in "
        f"{variant!r}{' with the recorded overrides' if overrides else ''}"
        for key, value in seen.items()
        if key in expected and _plain(value) != _plain(expected[key])
    ]
    if differing:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names the variant {variant!r} of {family!r} but "
            f"{model_file} is not that variant ({'; '.join(differing)}); the file and the model are "
            "not from one run"
        )


def _run_on_zeros(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> Any:
    """Return ``model`` called on one zero input of the record's shape, or raise :class:`ConfigError`.

    A model that cannot take the record's ``input_shape`` (another side, another channel count)
    fails inside Keras or TensorFlow with a message about a layer; here it is named as what it
    is, at load, not at the first prediction.
    """
    import tensorflow as tf

    try:
        return model(_zeros_of(record), training=False)
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        key = INPUT_KEY[record["task"]]
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: {model_file} does not take the {key} "
            f"{record[key]} it names ({type(exc).__name__}: {first[:150]}); the file "
            "and the model are not from one run"
        ) from exc


def _zeros_of(record: Mapping[str, Any]) -> Any:
    """Return one all-zero input of the record's shape (any rank), to ask a loaded model its output width."""
    import numpy as np

    return np.zeros((1, *input_shape_of(record)), np.float32)


def _check_width(logits: Any, record: Mapping[str, Any], unit: str = "image") -> None:
    """Raise :class:`ConfigError` unless ``logits`` is ``(N, classes)`` for the record's classes; ``unit`` names what a row is."""
    shape = tuple(logits.shape)
    if len(shape) != 2 or shape[1] != len(record["class_names"]):
        raise ConfigError(
            f"The model outputs {shape[1:]} per {unit} but {TASK_FILE} names "
            f"{len(record['class_names'])} classes; the file and the model are not from one run"
        )


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class Predictor:
    """A loaded model and the preprocessing it was trained with; see the module contract.

    :param model: the Keras model; its outputs are logits.
    :param record: a record that passed :func:`check_record`.
    """

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != SERVED_TASK:
            raise ConfigError(
                f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor "
                f"serves {SERVED_TASK!r} only"
            )
        self.model = model
        self.record = dict(record)
        self.class_names: List[str] = list(record["class_names"])

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def _to_source_channels(self, image: Any) -> Any:
        """Return ``image`` ``(H, W, C)`` with ``eval_transform.source_channels`` channels."""
        import tensorflow as tf

        wanted = self.record["eval_transform"]["source_channels"]
        if int(image.shape[-1]) == wanted:
            return image
        # Colour to grey is NOT done here: it happens on the 8-bit pixels in `_from_array`.
        return tf.repeat(image, wanted, axis=-1)

    def _from_array(self, array: Any) -> Any:
        """Return one array as a float32 ``[0, 1]`` tensor ``(H, W, source_channels)``."""
        import numpy as np
        import tensorflow as tf

        from train.common.classification_data import rgb8_to_grey8

        if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
            raise ConfigError(
                f"Invalid image: an array must be uint8 or floating point, got {array.dtype}"
            )
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4) or min(array.shape[:2]) < 1:
            raise ConfigError(
                "Invalid image: an array must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got "
                f"shape {tuple(array.shape)}"
            )
        if array.shape[-1] == 4:
            # DECISION plan-2026-10-30T132432-a865fa08/D-068: an RGBA array loses its alpha channel,
            # exactly as ``PIL.Image.convert("RGB")`` does for a PIL image. Do NOT composite it
            # over a background: the two input forms would then give different pixels.
            array = array[..., :3]
        if array.dtype != np.uint8 and not (np.all(array >= 0.0) and np.all(array <= 1.0)):
            raise ConfigError(
                "Invalid image: a floating array must lie in [0, 1] (the scale of a "
                "uint8 array divided by 255), got values outside it or NaN"
            )
        if array.shape[-1] == 3 and self.record["eval_transform"]["source_channels"] == 1:
            # DECISION plan-2026-10-31T180033-2faf890a/D-019
            # Grey is made on the 8-bit pixels with the loader's own integer rule, the stage
            # and the rounding at which `tf.io.decode_image(channels=1)` makes it for a PNG.
            # Do NOT use `tf.image.rgb_to_grayscale` (float luma, other weights: a level off
            # in a quarter of the pixels) or PIL's `convert("L")` (rounds: a level off in half).
            # A float array is rounded to the 8-bit value first, as storing it as a PNG would.
            # A JPEG already decoded to RGB cannot be exact: its Y plane is gone (module docstring).
            if array.dtype != np.uint8:
                array = np.rint(array.astype(np.float64) * 255.0).astype(np.uint8)
            array = rgb8_to_grey8(array)
        if array.dtype == np.uint8:
            image = tf.image.convert_image_dtype(tf.constant(array), tf.float32)
        else:
            image = tf.constant(array.astype(np.float32))
        return self._to_source_channels(image)

    def _from_path(self, path: Union[str, "os.PathLike[str]"]) -> Any:
        import tensorflow as tf

        from train.common.classification_data import decode_image_file

        if not os.path.isfile(path):
            raise ConfigError(f"Invalid image: {os.fspath(path)!r} is not a file")
        channels = self.record["eval_transform"]["source_channels"]
        try:
            return decode_image_file(os.fspath(path), channels)
        except tf.errors.InvalidArgumentError as exc:
            raise ConfigError(
                f"Invalid image: {os.fspath(path)!r} is not an image file tf.io.decode_image reads"
            ) from exc

    def _one(self, item: Any) -> Any:
        """Return one image (a path, a PIL image or a 2-D or 3-D array) as a tensor."""
        import numpy as np
        from PIL import Image

        if isinstance(item, (str, os.PathLike)):
            return self._from_path(item)
        if isinstance(item, Image.Image):
            if item.mode in WIDE_MODES:
                raise ConfigError(
                    f"Invalid image: a PIL image of mode {item.mode} holds 32-bit values and its "
                    "bit depth cannot be preserved (the loader reads no such file); convert it "
                    "to an 8-bit mode ('L' or 'RGB') yourself, or pass the file's path"
                )
            if item.mode in GREY16_MODES:
                # DECISION plan-2026-10-31T180033-2faf890a/D-028
                # A 16-bit grey image is read as `tf.io.decode_image` reads the file: the high
                # byte of each sample. Do NOT hand it to PIL's `convert`: that CLIPS to 255 and
                # was 254 levels off in 99% of the pixels (review-iter-1 concern 4).
                high = (np.asarray(item).astype(np.uint16) >> 8).astype(np.uint8)
                return self._from_array(high)
            # A colour image keeps its colours (alpha dropped) so `_from_array` converts them
            # to grey the loader's way; only an image that is already grey is read as grey.
            grey = self.record["eval_transform"]["source_channels"] == 1 and item.mode in GREY_MODES
            return self._from_array(np.asarray(item.convert("L" if grey else "RGB")))
        if isinstance(item, np.ndarray) and item.ndim in (2, 3):
            return self._from_array(item)
        raise ConfigError(
            "Invalid image: expected a path, a PIL image, or a 2-D or 3-D numpy array, got "
            f"{type(item).__name__}"
            + (f" of shape {tuple(item.shape)}" if hasattr(item, "shape") else "")
        )

    def _images_of(self, x: Any) -> Tuple[List[Any], bool]:
        """Return ``(images, several)`` for the argument of :meth:`predict`."""
        import numpy as np

        if isinstance(x, np.ndarray) and x.ndim == 4:
            if len(x) == 0:
                raise ConfigError(f"Invalid image: the array of images is empty (shape {tuple(x.shape)})")
            return [self._one(image) for image in x], True
        if isinstance(x, (list, tuple)):
            if not x:
                raise ConfigError("Invalid image: the list of images is empty")
            return [self._one(item) for item in x], True
        return [self._one(x)], False

    # -----------------------------------------------------------------
    # Preprocessing and scores
    # -----------------------------------------------------------------

    def _shaped(self, image: Any) -> Any:
        """Return ``image`` resized the way the evaluation pipeline did, ``(S, S, channels)``."""
        import tensorflow as tf

        from dl_techniques.datasets.vision.crops import resize_and_centre_crop
        from dl_techniques.datasets.vision.imagenet2012 import center_crop
        from train.common.classification_data import resizer

        side = self.record["input_shape"][0]
        kind = self.record["eval_transform"]["kind"]
        if kind == "resize":
            image = resizer(side)(image)
        elif kind == "resize_centre_crop":
            image = resize_and_centre_crop(image, side)
        else:
            image = center_crop(image, side)
        wanted = self.record["channels"]
        if int(image.shape[-1]) != wanted:
            image = tf.repeat(image, wanted, axis=-1)
        return image

    def probabilities(self, images: Sequence[Any]) -> Any:
        """Return the softmax probabilities ``(N, classes)`` of ``images`` (float64 ``numpy``).

        :param images: tensors from :meth:`_one`: float32 in ``[0, 1]``, any size.
        """
        import numpy as np
        import tensorflow as tf

        mean = tf.constant(np.asarray(self.record["mean"], np.float32))
        std = tf.constant(np.asarray(self.record["std"], np.float32))
        rows = []
        for start in range(0, len(images), CHUNK):
            batch = tf.stack([self._shaped(image) for image in images[start : start + CHUNK]])
            logits = self.model((batch - mean) / std, training=False)
            rows.append(np.asarray(logits, dtype=np.float64))
        logits = np.concatenate(rows, axis=0)
        _check_width(logits, self.record)
        return _softmax(logits)

    def predict(
        self, x: Any, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return ``{"labels", "scores"}`` for ``x``; see the module contract."""
        import numpy as np

        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ConfigError(f"Invalid value for 'top_k': expected a positive int or None, got {top_k!r}")
        if threshold is not None and (
            isinstance(threshold, bool) or not _is_number(threshold) or not 0 <= threshold <= 1
        ):
            raise ConfigError(
                f"Invalid value for 'threshold': expected a number in [0, 1] or None, got {threshold!r}"
            )
        images, several = self._images_of(x)
        probabilities = self.probabilities(images)
        labels: List[List[str]] = []
        scores: List[List[float]] = []
        for row in probabilities:
            order = np.argsort(-row, kind="stable")[:top_k]
            kept = [int(i) for i in order if threshold is None or row[i] >= threshold]
            labels.append([self.class_names[i] for i in kept])
            scores.append([float(row[i]) for i in kept])
        if several:
            return {"labels": labels, "scores": scores}
        return {"labels": labels[0], "scores": scores[0]}


# ---------------------------------------------------------------------
# The per-task tables (D-006). Each is keyed by the task word; ``RECORD_KEYS`` and ``DATA_KEYS``
# are above. A row is a callable that does ONE task's part of the work, and a task that is read,
# loaded and predicted with has a row in every table of ``TASK_TABLES``.
#
# Interface contract for a new task's rows:
#   GEOMETRY_CHECKS[task](record) -> (lines, channels or None): problems of the geometry keys, and
#       the channel count the statistics are checked against (``None`` when unusable);
#   TAIL_CHECKS[task](record) -> lines: problems of the keys only this task holds;
#   INPUT_SHAPE_OF[task](record) -> (height, width, channels); INPUT_KEY[task] -> the record key
#       an input-shape complaint names;
#   MODEL_CHECKS[task](model, record, model_file, task_file) -> None, raising ConfigError when the LOADED model is not
#       the one the record names by what the model says of itself (a forecaster's built window), BEFORE it is called;
#   OUTPUT_CHECKS[task](output, record) -> None, raising ConfigError when the model's output on one
#       zero input is not what the record names;
#   PREDICTORS[task](model, record) -> the object ``load_model`` returns;
#   EXPORT_RULES[task] -> an :class:`ExportRule` (the traced shape, the refused cells, the logged sentence).
# A row that imports TensorFlow does it inside the callable, never at module scope.
# ---------------------------------------------------------------------


def _segmentation_output_check(output: Any, record: Mapping[str, Any]) -> None:
    from .segmentation_predictor import check_output

    check_output(output, record)


def _denoising_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    """Return ``(side, side, channels)`` of the zero image a denoiser is called on (its own input is flexible)."""
    # DECISION plan-2026-11-10T095307-d483ffab/D-028
    # A denoising record has NO input size: the model is saved with a flexible input, and a size in the record would
    # be a number nothing reads (and a lie for every other size). Do NOT add ``input_size`` or ``input_shape`` to it.
    # The row returns a PROBE size, a multiple of the stride, so ``load_model`` can call the model once on zeros.
    multiple = record["size_multiple"]
    side = -(-DENOISING_PROBE_SIDE // multiple) * multiple
    return (side, side, record["channels"])


def _denoising_output_check(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output`` is ONE image tensor of the probe's size and the record's channels."""
    side, _, channels = _denoising_input_shape(record)
    shape = tuple(output.shape) if not isinstance(output, (list, tuple)) else None
    if shape != (1, side, side, channels):
        seen = f"{len(output)} outputs" if shape is None else str(shape)
        raise ConfigError(
            f"The model outputs {seen} for one {side}x{side}x{channels} image but {TASK_FILE} names a denoiser "
            f"of {channels} channels (one output of shape (1, {side}, {side}, {channels})); the file and the "
            "model are not from one run"
        )


def _no_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """The model check of a task whose model is held to its record by the family check and the zero-input run alone."""


def _forecasting_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    from .forecast_predictor import check_built_window

    check_built_window(model, record, model_file, task_file)


def _forecasting_output_check(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless every tensor of ``output`` holds one window (the batch of the zero window)."""
    leaves = output if isinstance(output, (list, tuple)) else [output]
    seen = [tuple(leaf.shape) for leaf in leaves]
    if not seen or any(not shape or shape[0] != 1 for shape in seen):
        raise ConfigError(
            f"The model outputs {seen} for one window of {record['lookback']} rows but {TASK_FILE} names a forecaster "
            "(every output has the batch of its input first); the file and the model are not from one run"
        )


def _depth_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    from .depth_predictor import check_built_size

    check_built_size(model, record, model_file, task_file)


def _depth_output_check(output: Any, record: Mapping[str, Any]) -> None:
    from .depth_predictor import check_output

    check_output(output, record)


def _pretrain_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    from .pretrain_predictor import check_built_size

    check_built_size(model, record, model_file, task_file)


def _pretrain_output_check(output: Any, record: Mapping[str, Any]) -> None:
    from .pretrain_predictor import check_output

    check_output(output, record)


def _dino_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    from .dino_predictor import check_built_model

    check_built_model(model, record, model_file, task_file)


def _dino_output_check(output: Any, record: Mapping[str, Any]) -> None:
    from .dino_predictor import check_output

    check_output(output, record)


def _tokenizer_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    from .tokenizer_predictor import check_built_model

    check_built_model(model, record, model_file, task_file)


def _tokenizer_output_check(output: Any, record: Mapping[str, Any]) -> None:
    from .tokenizer_predictor import check_output

    check_output(output, record)


def _language_model_output_check(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output`` is the logits ``(1, seq_len, vocab_size)`` of the zero document."""
    wanted = (1, record["seq_len"], record["vocab_size"])
    shape = tuple(output.shape) if not isinstance(output, (list, tuple, dict)) else None
    if shape != wanted:
        seen = f"{type(output).__name__} of {len(output)} outputs" if shape is None else str(shape[1:])
        raise ConfigError(
            f"The model outputs {seen} for one document of {record['seq_len']} ids but {TASK_FILE} names a vocabulary of "
            f"{record['vocab_size']} ({wanted[1:]}); the file and the model are not from one run"
        )


def _text_classification_model_check(model: Any, record: Mapping[str, Any], model_file: Path, task_file: Path) -> None:
    """Raise :class:`ConfigError` unless the embedding of ``model`` has the ``vocab_size`` of the record (its logits are class scores, which show none)."""
    # DECISION plan-2026-11-24T092348-7806a290/D-062
    # The language model's logits are as wide as its vocabulary, so ``_language_model_output_check`` sees a record of another one; a
    # classifier's are not, so its vocabulary is read off the backbone layer here. Do NOT add ``vocab_size`` to the family's ``observe``
    # (its keys are compared with the variant's table, which has none) and do NOT drop this row: an id beyond the table raises
    # TensorFlow's ``InvalidArgumentError`` on a CPU (measured) from ``predict`` and ``export`` instead of a named refusal at load.
    from .text_models import vocabulary_of

    held = vocabulary_of(model)
    if held is not None and held != record["vocab_size"]:
        raise ConfigError(
            f"Invalid {TASK_FILE} {task_file}: it names a vocabulary of {record['vocab_size']} ids but {model_file} embeds {held}; the "
            "file and the model are not from one run"
        )


def _text_classification_output_check(output: Any, record: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` unless ``output`` is the logits ``(1, classes)`` of the zero document."""
    _check_width(output, record, "document")


def _classification_predictor(model: Any, record: Mapping[str, Any]) -> "Predictor":
    return Predictor(model, record)


def _segmentation_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .segmentation_predictor import SegmentationPredictor

    return SegmentationPredictor(model, record)


def _denoising_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .denoising_predictor import DenoisingPredictor

    return DenoisingPredictor(model, record)


def _forecasting_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .forecast_predictor import ForecastPredictor

    return ForecastPredictor(model, record)


def _depth_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .depth_predictor import DepthPredictor

    return DepthPredictor(model, record)


def _pretrain_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .pretrain_predictor import PretrainPredictor

    return PretrainPredictor(model, record)


def _dino_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .dino_predictor import DinoPredictor

    return DinoPredictor(model, record)


def _tokenizer_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .tokenizer_predictor import TokenizerPredictor

    return TokenizerPredictor(model, record)


def _language_model_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .text_predictor import LanguageModelPredictor

    return LanguageModelPredictor(model, record)


def _text_classification_predictor(model: Any, record: Mapping[str, Any]) -> Any:
    from .text_predictor import TextClassifierPredictor

    return TextClassifierPredictor(model, record)


class ExportRule(NamedTuple):
    """One task's part of ``export``, a row of :data:`EXPORT_RULES`.

    :param shape_of: ``f(record) -> (height, width, channels)`` the command line traces the export with; a side
        is ``None`` when the model takes any size (``export`` accepts that only when the model's own input does).
    :param check: ``f(record, format, shape) -> None`` raising :class:`ConfigError` for a cell of the task's
        export table that is refused; it runs before anything is written. ``shape`` is what the export traces.
    :param note: ``f(record) -> str``, the sentence the command line logs beside the exported model.
    :param served: ``None`` (the default: the exported function is ``model(batch, training=False)``, the three image
        tasks) or ``f(model, record) -> layer``, a callable Keras layer that IS the exported function: the
        saved_model endpoint, the tflite and onnx graph, the tflite numeric check and the ``keras`` archive all run it.
        It returns a tensor, or a dict holding ``point`` (tflite and onnx take that one); its ``static_batch``
        attribute, when not ``None``, is the batch the file is traced at. The length of ``shape_of(record)`` is the
        rank ``export`` accepts of ``input_shape`` (3 for the image tasks, 2 for ``(lookback, channels)``, 1 for ``(seq_len,)``).
    :param sample: ``None`` (the default) or ``f(record, shape) -> numpy float32`` batch of ``shape`` (``(batch, *input shape)``): the
        batch the tflite check runs the file and the served function on, in place of the random floats in [0, 1) every task was checked
        with before this field existed. A task whose input is whole numbers (a language model's token ids, which such floats all make
        id 0) names its own; it is bound to the record by ``export`` (D-008).
    """

    shape_of: Callable[[Mapping[str, Any]], Tuple[Optional[int], ...]]
    check: Callable[[Mapping[str, Any], str, Tuple[Optional[int], ...]], None]
    note: Callable[[Mapping[str, Any]], str]
    served: Optional[Callable[[Any, Mapping[str, Any]], Any]] = None
    sample: Optional[Callable[[Mapping[str, Any], Tuple[int, ...]], Any]] = None


def _no_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    """The export rule of a task whose every cell exports: nothing is refused."""


#: Why a tflite file of an Energy Transformer is refused by both of its tasks (pretrain ``et_mim``, plan 5c6cc105 D-024;
#: classification ``energy_transformer``, D-030): the converter's error, measured on both.
HOPFIELD_TFLITE_CAUSE = (
    "the TFLite converter of TensorFlow 2.18 mistranslates the Hopfield update of every energy-descent step "
    "(einsum, relu, einsum), so the file is about 0.07 of the scale off after one step and 0.56 after 12, with "
    "built-in and with select operators alike"
)

# DECISION plan-2026-11-19T081214-5c6cc105/D-030
# The energy_transformer tflite cell is refused HERE, before anything is written. Do NOT let the export try it: a
# classifier has no served function, so the built-in file is written without a numeric check, and it is wrong (1.46 off,
# 0.53 of the scale, on et_tiny). `test_the_energy_transformer_tflite_refusal_still_has_its_reason` says when to lift it.
#: The classification export cells that are refused, ``(family, format) -> reason``, each measured.
CLASSIFICATION_EXPORT_REFUSALS: Dict[Tuple[str, str], str] = {
    (ET_CLASSIFIER, "tflite"): (
        f"{HOPFIELD_TFLITE_CAUSE}, and a classifier's built-in file is written with no numeric check, so it would be "
        "wrong without a word (0.53 of the scale on et_tiny); choose keras, saved_model or onnx"
    ),
}


def refuse_export_cell(refusals: Mapping[Tuple[str, str], str], record: Mapping[str, Any], format: str) -> None:
    """Raise :class:`ConfigError` when ``(record["family"], format)`` is a row of ``refusals``, quoting its reason.

    The refusal of the classification and pretrain export checks; it writes nothing.
    """
    reason = refusals.get((record["family"], format))
    if reason is not None:
        raise ConfigError(
            f"Invalid value for 'format': {format!r} of a model of the {record['family']!r} family is refused: {reason}"
        )


def _classification_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    refuse_export_cell(CLASSIFICATION_EXPORT_REFUSALS, record, format)


def _standardized_logits_note(record: Mapping[str, Any]) -> str:
    return (
        "the exported model takes inputs standardized with the run's mean and std (pixels in [0, 1], then "
        "(x - mean) / std) and returns raw logits"
    )


def _denoising_export_shape(record: Mapping[str, Any]) -> Tuple[Optional[int], ...]:
    # DECISION plan-2026-11-10T095307-d483ffab/D-031
    # A denoiser is exported with a FREE height and width. Do NOT trace it at the probe size (the shape of
    # ``INPUT_SHAPE_OF``): a fixed-shape tflite of a U-Net returns wrong numbers at any other size WITHOUT an error
    # (measured, step 2(b)), and the record holds no size to fix it at (D-028).
    return (None, None, record["channels"])


def _denoising_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .denoising_predictor import check_export

    check_export(record, format, shape)


def _denoising_export_note(record: Mapping[str, Any]) -> str:
    from .denoising_predictor import export_note

    return export_note(record)


def _forecast_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .forecast_predictor import check_export

    check_export(record, format, shape)


def _forecast_served(model: Any, record: Mapping[str, Any]) -> Any:
    from .forecast_predictor import served

    return served(model, record)


def _forecast_export_note(record: Mapping[str, Any]) -> str:
    from .forecast_predictor import export_note

    return export_note(record)


def _depth_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .depth_predictor import check_export

    check_export(record, format, shape)


def _depth_export_note(record: Mapping[str, Any]) -> str:
    from .depth_predictor import export_note

    return export_note(record)


def _pretrain_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .pretrain_predictor import check_export

    check_export(record, format, shape)


def _pretrain_served(model: Any, record: Mapping[str, Any]) -> Any:
    from .pretrain_predictor import served

    return served(model, record)


def _pretrain_export_note(record: Mapping[str, Any]) -> str:
    from .pretrain_predictor import export_note

    return export_note(record)


def _dino_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .dino_predictor import check_export

    check_export(record, format, shape)


def _dino_served(model: Any, record: Mapping[str, Any]) -> Any:
    from .dino_predictor import served

    return served(model, record)


def _dino_export_note(record: Mapping[str, Any]) -> str:
    from .dino_predictor import export_note

    return export_note(record)


def _tokenizer_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .tokenizer_predictor import check_export

    check_export(record, format, shape)


def _tokenizer_served(model: Any, record: Mapping[str, Any]) -> Any:
    from .tokenizer_predictor import served

    return served(model, record)


def _tokenizer_export_note(record: Mapping[str, Any]) -> str:
    from .tokenizer_predictor import export_note

    return export_note(record)


def _text_export_check(record: Mapping[str, Any], format: str, shape: Tuple[Optional[int], ...]) -> None:
    from .text_predictor import check_export

    check_export(record, format, shape)


def _text_served(model: Any, record: Mapping[str, Any]) -> Any:
    from .text_predictor import served

    return served(model, record)


def _text_sample(record: Mapping[str, Any], shape: Tuple[int, ...]) -> Any:
    from .text_predictor import sample_ids

    return sample_ids(record, shape)


def _language_model_export_note(record: Mapping[str, Any]) -> str:
    from .text_predictor import export_note

    return export_note(record)


def _text_classification_export_note(record: Mapping[str, Any]) -> str:
    from .text_predictor import classifier_export_note

    return classifier_export_note(record)


def _text_input_shape(record: Mapping[str, Any]) -> Tuple[int, ...]:
    return (record["seq_len"],)


#: Geometry keys and the channel count, per task (see the contract above).
GEOMETRY_CHECKS: Dict[str, Callable[[Mapping[str, Any]], Tuple[List[str], Optional[int]]]] = {
    CLASSIFICATION: _classification_geometry,
    SEGMENTATION: _segmentation_geometry,
    DENOISING: _denoising_geometry,
    FORECASTING: _forecast_geometry,
    DEPTH: _depth_geometry,
    PRETRAIN: _pretrain_geometry,
    LANGUAGE_MODEL: _language_model_geometry,
    TEXT_CLASSIFICATION: _text_classification_geometry,
    BEIT_TOKENIZER: _pretrain_geometry,
    DINO: _pretrain_geometry,
}

#: The keys only one task holds (classification's ``eval_transform``, segmentation's void,
#: threshold and ``output``, denoising's ``data_range`` and ``output``), per task.
TAIL_CHECKS: Dict[str, Callable[[Mapping[str, Any]], List[str]]] = {
    CLASSIFICATION: _eval_transform_problems,
    SEGMENTATION: _segmentation_tail_problems,
    DENOISING: _denoising_tail_problems,
    FORECASTING: _forecast_tail_problems,
    DEPTH: _depth_tail_problems,
    PRETRAIN: _pretrain_tail_problems,
    LANGUAGE_MODEL: _language_model_tail_problems,
    TEXT_CLASSIFICATION: _text_classification_tail_problems,
    BEIT_TOKENIZER: _tokenizer_tail_problems,
    DINO: _dino_tail_problems,
}

#: A denoiser's own input is flexible, so its row gives the zero image it is probed with.
INPUT_SHAPE_OF: Dict[str, Callable[[Mapping[str, Any]], Tuple[int, ...]]] = {
    CLASSIFICATION: _classification_input_shape,
    SEGMENTATION: _segmentation_input_shape,
    DENOISING: _denoising_input_shape,
    FORECASTING: _forecast_input_shape,
    DEPTH: _segmentation_input_shape,
    PRETRAIN: _segmentation_input_shape,
    LANGUAGE_MODEL: _text_input_shape,
    TEXT_CLASSIFICATION: _text_input_shape,
    BEIT_TOKENIZER: _segmentation_input_shape,
    DINO: _dino_input_shape,
}

#: The record key that holds the input geometry, per task (a denoiser's is its channel count).
INPUT_KEY: Dict[str, str] = {
    CLASSIFICATION: "input_shape", SEGMENTATION: "input_size", DENOISING: "channels", FORECASTING: "lookback",
    DEPTH: "input_size", PRETRAIN: "input_size", LANGUAGE_MODEL: "seq_len", TEXT_CLASSIFICATION: "seq_len", BEIT_TOKENIZER: "input_size",
    DINO: "input_size",
}

#: What a loaded model says of itself against its record, run BEFORE the model is called (D-042): a forecaster's built
#: window, a depth model's built size and a pretrain backbone's built size and channels.
MODEL_CHECKS: Dict[str, Callable[[Any, Mapping[str, Any], Path, Path], None]] = {
    CLASSIFICATION: _no_model_check,
    SEGMENTATION: _no_model_check,
    DENOISING: _no_model_check,
    FORECASTING: _forecasting_model_check,
    DEPTH: _depth_model_check,
    PRETRAIN: _pretrain_model_check,
    # A functional model takes one static length: the zero-document call of ``load_model`` refuses another one by name (``INPUT_KEY``).
    LANGUAGE_MODEL: _no_model_check,
    # A classifier's output is class scores, which show no vocabulary: the embedding's rows are read off the backbone layer instead.
    TEXT_CLASSIFICATION: _text_classification_model_check,
    # A tokenizer's size, patch and codebook are held to the model's own config, and its code grid is proved by running it on a zero image
    # (the output of ``load_model``'s own call is the RECONSTRUCTION, which shows no grid): D-044.
    BEIT_TOKENIZER: _tokenizer_model_check,
    # A dino file is the TRAINING model: its view count, size, patch, head and embedding widths are held to the record before the model is called
    # on a view set the record could make fail (D-028).
    DINO: _dino_model_check,
}

OUTPUT_CHECKS: Dict[str, Callable[[Any, Mapping[str, Any]], None]] = {
    CLASSIFICATION: _check_width,
    SEGMENTATION: _segmentation_output_check,
    DENOISING: _denoising_output_check,
    FORECASTING: _forecasting_output_check,
    DEPTH: _depth_output_check,
    PRETRAIN: _pretrain_output_check,
    LANGUAGE_MODEL: _language_model_output_check,
    TEXT_CLASSIFICATION: _text_classification_output_check,
    BEIT_TOKENIZER: _tokenizer_output_check,
    DINO: _dino_output_check,
}

PREDICTORS: Dict[str, Callable[[Any, Mapping[str, Any]], Any]] = {
    CLASSIFICATION: _classification_predictor,
    SEGMENTATION: _segmentation_predictor,
    DENOISING: _denoising_predictor,
    FORECASTING: _forecasting_predictor,
    DEPTH: _depth_predictor,
    PRETRAIN: _pretrain_predictor,
    LANGUAGE_MODEL: _language_model_predictor,
    TEXT_CLASSIFICATION: _text_classification_predictor,
    BEIT_TOKENIZER: _tokenizer_predictor,
    DINO: _dino_predictor,
}

#: What ``export`` does per task (see :class:`ExportRule`). Classification and segmentation trace at the record's
#: own input shape and say the exported model takes standardized inputs and returns logits; segmentation refuses no
#: cell, classification the cells of :data:`CLASSIFICATION_EXPORT_REFUSALS`.
EXPORT_RULES: Dict[str, ExportRule] = {
    CLASSIFICATION: ExportRule(_classification_input_shape, _classification_export_check, _standardized_logits_note),
    SEGMENTATION: ExportRule(_segmentation_input_shape, _no_export_check, _standardized_logits_note),
    DENOISING: ExportRule(_denoising_export_shape, _denoising_export_check, _denoising_export_note),
    # A forecaster's file takes RAW windows ``(lookback, channels)`` (rank 2) and returns a forecast in the series' units: it
    # runs ``served``; the cells of its table that do not convert are refused by ``check`` (D-011, D-026).
    FORECASTING: ExportRule(_forecast_input_shape, _forecast_export_check, _forecast_export_note, _forecast_served),
    # A depth model's file takes ImageNet-standardized images at the trained size (D-009): the record's own input shape;
    # every format is written, and ``check`` refuses any other size by name (D-018).
    DEPTH: ExportRule(_segmentation_input_shape, _depth_export_check, _depth_export_note),
    # A pretrain model's file is its backbone's embeddings at the trained size: ``served`` is a functional model over
    # the run's backbone, ``check`` refuses another size and the measured cells that do not convert (plan 5c6cc105
    # D-003, D-024).
    PRETRAIN: ExportRule(_segmentation_input_shape, _pretrain_export_check, _pretrain_export_note, _pretrain_served),
    # A language model's file takes token ids as float32 ``(seq_len,)`` at the trained length and returns the next-token logits: ``served``
    # is the model with ``training=False`` (a rule with none is written with no tflite check), ``sample`` the seeded ids the check is
    # fed (plan 7806a290 D-008, D-032). Every cell converts (``test_language_model_pretraining_predictor_and_export.py``), so ``check`` refuses a length only.
    LANGUAGE_MODEL: ExportRule(
        _text_input_shape, _text_export_check, _language_model_export_note, _text_served,
        _text_sample),
    # DECISION plan-2026-11-24T092348-7806a290/D-063
    # A text classifier's file is the language model's rule with its own note: float ids at the trained length in, class logits out. The
    # shape, the check, the served function and the sample read only ``seq_len``, ``vocab_size`` and ``pad_id``, which its record holds too.
    # Do NOT give it ``served=None`` or no ``sample`` (the built-in tflite file would then be written with no numeric check, or checked on
    # random floats that are all id 0), and do NOT refuse a cell: every one converts and matches on valid rows (measured, D-063).
    TEXT_CLASSIFICATION: ExportRule(
        _text_input_shape, _text_export_check, _text_classification_export_note, _text_served,
        _text_sample),
    # DECISION plan-2026-11-25T124419-fa94f59c/D-046
    # A tokenizer's file takes standardized images at the trained size and returns the int32 code grid: `served` is the registered layer
    # around the model, `check` refuses another size, and NO cell is refused (4 formats x 4 models measured: each file gave Keras's code at
    # every one of 4,096 and of 32,768 positions; `scratch_export/measured.md`). Do NOT add a refusal table "for near ties": the limit is
    # the argmin's, at positions whose two nearest codes are within float32 rounding; it holds for every format, and the export note says
    # so. Do NOT add a `sample`: the shared check batch reaches 6 to 32 codes, and a mistranslated op of this graph (a wrong axis, a
    # transposed grid) changes nearly every one of its 128 positions (a test plants one).
    BEIT_TOKENIZER: ExportRule(_segmentation_input_shape, _tokenizer_export_check, _tokenizer_export_note, _tokenizer_served),
    # DECISION plan-2026-11-26T224507-e3381378/D-028
    # A dino export is the DERIVED feature model, image to CLS embedding, at the trained size (the loaded file is the training model, which takes
    # a view set): `shape_of` is the record's `(side, side, channels)`, NOT `INPUT_SHAPE_OF`'s rank-5 view set; `served` is the functional student
    # feature model; `check` refuses another size and no cell (P6: keras, saved_model, tflite and onnx all converted and matched, 8 of 8). Do NOT
    # reuse `INPUT_SHAPE_OF` here and do NOT export the training model: it holds a teacher, a head and a loss-shaped output.
    DINO: ExportRule(_segmentation_input_shape, _dino_export_check, _dino_export_note, _dino_served),
}

#: Every per-task table by name. A task is KNOWN to this reader only when it has a row in all of
#: them; ``record_problems`` refuses any other, naming the tasks of the first table it is absent
#: from. ``tests/test_train/test_api/test_a_third_task_record_dispatches_by_table.py`` walks this.
TASK_TABLES: Dict[str, Mapping[str, Any]] = {
    "RECORD_KEYS": RECORD_KEYS,
    "DATA_KEYS": DATA_KEYS,
    "GEOMETRY_CHECKS": GEOMETRY_CHECKS,
    "TAIL_CHECKS": TAIL_CHECKS,
    "INPUT_SHAPE_OF": INPUT_SHAPE_OF,
    "INPUT_KEY": INPUT_KEY,
    "MODEL_CHECKS": MODEL_CHECKS,
    "OUTPUT_CHECKS": OUTPUT_CHECKS,
    "PREDICTORS": PREDICTORS,
    "EXPORT_RULES": EXPORT_RULES,
}
