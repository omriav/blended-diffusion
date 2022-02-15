from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments


if __name__ == "__main__":
    args = get_arguments()
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_prompt()
    # image_editor.reconstruct_image()
