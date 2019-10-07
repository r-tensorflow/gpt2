#' @export
install_gtp2 <- function(method = c("auto", "virtualenv", "conda"),
                         conda = "auto",
                         tensorflow = "default",
                         ...) {

  # verify method

  # some special handling for windows
  if (identical(.Platform$OS.type, "windows")) {

    # conda is the only supported method on windows
    method <- "conda"

    # confirm we actually have conda
    have_conda <- !is.null(tryCatch(reticulate::conda_binary(conda), error = function(e) NULL))
    if (!have_conda) {
      stop("GPT-2 installation failed (no conda binary found)\n\n",
           "Install Anaconda for Python 3.x (https://www.anaconda.com/download/#windows)\n",
           "before installing GPT-2",
           call. = FALSE)
    }

    # avoid DLL in use errors
    if (reticulate::py_available()) {
      stop("You should call install_gtp2() only in a fresh ",
           "R session that has not yet initialized GPT-2 and TensorFlow (this is ",
           "to avoid DLL in use errors during installation)")
    }
  }

  extra_packages <- unique(c(
    "fire>=0.1.3",
    "regex>=2017.4.5",
    "requests>=2.21.0",
    "tqdm>=4.31.1"
  ))

  # perform the install
  tensorflow::install_tensorflow(method = method,
                                 conda = conda,
                                 version = tensorflow,
                                 extra_packages = extra_packages,
                                 pip_ignore_installed = FALSE,
                                 ...)
}
