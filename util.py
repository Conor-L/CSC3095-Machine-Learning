import tf.keras.preprocessing.image.ImageDataGenerator

def flow_from_directory(image_dataset: ImageDataGenerator, subset: str):
    return image_dataset.flow_from_directory(batch_size=config['batch_size'], 
                                               directory=total_directory,
                                               target_size=(256, 256),
                                               class_mode='categorical',
                                               shuffle=True,
                                               color_mode='grayscale',
                                               subset=subset)