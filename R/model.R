#' @export
gtp2 <- function(model = c("124M", "355M", "774M")) {
  model <- match.arg(model)
  pin_name <- paste("gpt2", model, sep = "_")

  if (nrow(pins::pin_find(name = pin_name) == 0)) install_gtp2(model = model)


  py_path <- system.file("python", package = "gpt2")
  py_gpt2 <- import_from_path("gpt2", path = python_path)

  NULL
}
