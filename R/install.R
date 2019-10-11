install_gpt2_deps <- function() {
  c(
    "fire",
    "regex",
    "requests",
    "tqdm"
  )
}

install_gpt2_python_version <- function() {
  sys <- reticulate::import("sys")
  sys$version_info$major
}

install_gpt2_verify <- function() {
  installed <- sapply(install_gpt2_deps(), function(e) reticulate::py_module_available(e))

  if (!all(installed)) stop("GTP-2 dependencies are missing, considere running install_gpt2().")

  sys <- reticulate::import("sys")
  if (install_gpt2_python_version() <= 2) {
    stop("Python 3 required, but Python ", sys$version_info$major, ".", sys$version_info$minor, " is installed.")
  }
}

#' @export
install_gpt2 <- function(method = c("auto", "virtualenv", "conda"),
                         conda = "auto",
                         envname = NULL,
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
      stop("You should call install_gpt2() only in a fresh ",
           "R session that has not yet initialized GPT-2 and TensorFlow (this is ",
           "to avoid DLL in use errors during installation)")
    }
  }

  extra_packages <- unique(install_gpt2_deps())

  # perform the install
  tensorflow::install_tensorflow(method = method,
                                 conda = conda,
                                 version = "1.12",
                                 extra_packages = extra_packages,
                                 pip_ignore_installed = FALSE,
                                 envname = envname,
                                 ...)
}
