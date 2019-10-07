#' @export
gpt2_download <- function(model = c("124M", "355M", "774M")) {
  model <- match.arg(model, c("124M", "355M", "774M"))

  model_base <- paste0("https://storage.googleapis.com/gpt-2/models/", model, "/")
  model_files <- c("checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta", "vocab.bpe")
  model_urls <- paste0(model_base, model_files)

  pins::pin(model_urls, name = paste("gpt2", model, sep = "_"), board = "local")
}
