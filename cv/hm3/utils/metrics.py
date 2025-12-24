import numpy as np

def ssd(template, template_init):
    assert template_init.shape == template.shape

    return -np.sum((template_init - template)**2)

def ncc(template, template_init):
    assert template_init.shape == template.shape

    template_init_norm = (template_init - template_init.mean()) / np.sqrt(np.sum(template_init**2))
    template_norm = (template - template.mean()) / np.sqrt(np.sum(template**2))
    corr = template_init_norm * template_norm

    return corr.sum()

def sad(template, template_init):
    assert template_init.shape == template.shape

    return -np.abs(template_init - template).sum()

def normalize(image):

    result = (image - image.min()) / (image.max() - image.min())
    result = 255 * result
    result = result.astype(np.uint8)

    return result
