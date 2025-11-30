# box::use(
#     # ./renv/activate,
#     box.lsp[box_use_parser]
# )

# activate$action

if (nzchar(system.file(package = "box.lsp"))) {
    options(
        languageserver.parser_hooks = list(
            "box::use" = box.lsp::box_use_parser
        )
    )
}
