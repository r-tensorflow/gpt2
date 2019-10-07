#' @export
gtp2 <- function(model = c("124M", "355M", "774M")) {
  model <- match.arg(model)
  pin_name <- paste("gpt2", model, sep = "_")

  if (nrow(pins::pin_find(name = pin_name) == 0)) install_gtp2(model = model)

  NULL
}
