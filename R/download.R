#' Download Model
#'
#' Downloads the pretrained models by size.
#'
#' @param model The size of the model to download: \code{"124M"}, \code{"355M"},
#'   \code{"774M"} or \code{"1558M"}.
#'
#' @details
#'
#' Download and caching is performed using the \code{pins} package and creates
#' pins prefixed with \code{gpt2_}.
#'
#' @export
gpt2_download <- function(model = c("124M", "355M", "774M", "1558M")) {
  model <- match.arg(model, c("124M", "355M", "774M", "1558M"))

  model_base <- paste0("https://storage.googleapis.com/gpt-2/models/", model, "/")
  model_files <- c("checkpoint", "encoder.json", "hparams.json", "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta", "vocab.bpe")
  model_urls <- paste0(model_base, model_files)

  pins::pin(model_urls, name = paste("gpt2", model, sep = "_"), board = "local")
}
