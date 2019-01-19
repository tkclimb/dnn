if(BUILD_DOCS)
  set(DOCS_DIR ${ROOT_DIR}/docs)

  # check if Doxygen is installed
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOCS_DOXYFILE_PATH ${DOCS_DIR}/Doxyfile)
    set(DOCS_DOXYGEN_DIR ${DOCS_DIR}/doxygen)

    if(EXISTS ${DOCS_DOXYGEN_DIR})
      file(REMOVE_RECURSE ${DOCS_DOXYGEN_DIR})
    endif()

    file(MAKE_DIRECTORY ${DOCS_DOXYGEN_DIR})

    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOCS_DOXYFILE_PATH}
        WORKING_DIRECTORY ${ROOT_DIR}
        COMMENT "Generating C++ document by doxygen"
        VERBATIM)

  else()
    message(FATAL_ERROR "Doxygen is not installed in your environment...")
  endif()
endif()
