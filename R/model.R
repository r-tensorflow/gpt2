#' @export
gpt2 <- function(model = c("124M", "355M", "774M")) {
  model <- match.arg(model)
  install_gpt2_verify()

  pin_name <- paste("gpt2", model, sep = "_")
  if (nrow(pins::pin_find(name = pin_name, board = "local")) == 0) gpt2_download(model = model)

  py_path <- system.file("python", package = "gpt2")
  py_gpt2 <- reticulate::import_from_path("gpt2", path = py_path)

  model_path <- dirname(dirname(pins::pin_get(name = pin_name, board = "local")[1]))
  py_gpt2$encoder$get_encoder(pin_name, model_path)

  NULL
}
