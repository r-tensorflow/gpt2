#' @importFrom reticulate %as%
#' @export
gpt2 <- function(prompt = "Hello my name is",
                 model = c("124M", "355M", "774M"),
                 temperature = 1,
                 top_k = 0) {
  model <- match.arg(model, choices = c("124M", "355M", "774M"))
  install_gpt2_verify()

  batch_size <- 1

  pin_name <- paste("gpt2", model, sep = "_")
  if (nrow(pins::pin_find(name = pin_name, board = "local")) == 0) gpt2_download(model = model)

  py_path <- system.file("python", package = "gpt2")
  py_gpt2 <- reticulate::import_from_path("gpt2", path = py_path)

  model_path <- dirname(dirname(pins::pin_get(name = pin_name, board = "local")[1]))
  encoder <- py_gpt2$encoder$get_encoder(pin_name, model_path)
  gtp2 <- py_gpt2$gtp2

  hparams <- gtp2$default_hparams()

  hparams_json <- paste0(readLines(file.path(model_path, pin_name, "hparams.json")), collapse = "\n")
  json <- reticulate::import("json")
  hparams_dict <- json$loads(hparams_json)
  hparams$override_from_dict(hparams_dict)

  hparams_length <- hparams$n_ctx

  tf <- tensorflow::tf
  with(tf$Session(graph = tf$Graph()) %as% sess, {
    context <- tf$placeholder(tf$int32, list(batch_size, NULL))

    context_tokens <- encoder$encode(prompt)

    output <- gtp2$sample_sequence(
      hparams = hparams,
      length = min(hparams_length, 1023 - length(context_tokens)),
      context = context,
      batch_size = batch_size,
      temperature = temperature,
      top_k = top_k
    )

    saver <- tf$train$Saver()
    ckpt <- tf$train$latest_checkpoint(file.path(model_path, pin_name))
    saver$restore(sess, ckpt)

    out <- sess$run(output, feed_dict = reticulate::dict(
      context = list(context_tokens)
    ))

    encoder$decode(out[1:nrow(out), (length(context_tokens)+1):ncol(out)])
  })
}
