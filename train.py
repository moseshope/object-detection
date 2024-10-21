import json
import json
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import shape, box
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import logging
import geopandas as gpd
import pyproj
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define projection transformer (Web Mercator <-> WGS84)
wgs84_to_webmercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
webmercator_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def validate_image(tile_image_path):
    """Validate if the image file is not corrupted."""
    try:
        with Image.open(tile_image_path) as img:
            img.verify()  # Verify that the image is not corrupted
        logging.debug(f"Image validated successfully: {tile_image_path}")
        return True
    except (UnidentifiedImageError, IOError) as e:
        logging.error(f"Cannot identify image file: {tile_image_path}. Error: {str(e)}")
        return False

def load_geojson_as_raw(file_path):
    """Load GeoJSON file as raw JSON to manually handle multipart geometries."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.debug(f"Successfully loaded GeoJSON from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading GeoJSON file as raw JSON: {str(e)}")
        return None

def extract_geometries_and_categories(geojson_raw):
    """Extract geometries and categories manually from the raw GeoJSON."""
    geometries = []
    categories = []

    for feature in geojson_raw["features"]:
        geometry = feature["geometry"]
        properties = feature["properties"]

        if geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                geometries.append(shape({"type": "Polygon", "coordinates": polygon}))
                categories.append(properties["Category"])
        elif geometry["type"] == "Polygon":
            geometries.append(shape(geometry))
            categories.append(properties["Category"])
        else:
            logging.warning(f"Unsupported geometry type: {geometry['type']}")

    logging.debug(f"Extracted {len(geometries)} geometries and {len(categories)} categories")
    logging.debug(f"Unique categories: {set(categories)}")
    return geometries, categories

def create_mask_from_geometries(geometries, categories, image_shape, bounds, category):
    """Create a binary mask from geometries for a specific category."""
    logging.debug(f"Creating mask for category: {category} with bounds: {bounds}")

    # Convert bounds from Web Mercator to WGS84 for comparison with GeoJSON features
    lon_min, lat_min = webmercator_to_wgs84.transform(bounds[0], bounds[1])
    lon_max, lat_max = webmercator_to_wgs84.transform(bounds[2], bounds[3])
    transformed_bounds = (lon_min, lat_min, lon_max, lat_max)
    logging.debug(f"Transformed tile bounds: {transformed_bounds}")

    # Select geometries that intersect with the transformed bounds
    selected_geometries = [g for g, c in zip(geometries, categories) if c == category and g.intersects(box(*transformed_bounds))]

    logging.debug(f"Found {len(selected_geometries)} geometries for category {category} within bounds")

    if not selected_geometries:
        logging.warning(f"No geometries found for category {category} within bounds {transformed_bounds}")
        return None

    try:
        mask = rasterize(
            [(geom, 1) for geom in selected_geometries],
            out_shape=image_shape,
            fill=0,
            default_value=1,
            dtype='uint8',
            all_touched=True,
            transform=rasterio.transform.from_bounds(*bounds, *image_shape)
        )
        logging.debug(f"Mask created with shape {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
        return mask
    except ValueError as e:
        logging.error(f"Rasterization error: {str(e)}")
        return None

def parse_filename(filename):
    """Parse filename to extract coordinates and zoom level"""
    parts = filename.split('_')
    try:
        x, y = int(parts[-3]), int(parts[-2])
        zoom = int(parts[-1].split('.')[0])
        return x, y, zoom
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing filename {filename}: {str(e)}")
        return None, None, None

def get_tile_bounds(tile_x, tile_y, zoom):
    """Get the geographic bounds of a tile given its x, y coordinates and zoom level."""
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lat_min = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (tile_y + 1) / n))))
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_max = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * tile_y / n))))

    # Transform lon/lat to Web Mercator
    x_min, y_min = wgs84_to_webmercator.transform(lon_min, lat_min)
    x_max, y_max = wgs84_to_webmercator.transform(lon_max, lat_max)
    bounds = (x_min, y_min, x_max, y_max)
    logging.debug(f"Calculated bounds for tile ({tile_x}, {tile_y}, zoom {zoom}): {bounds}")
    return bounds

def preprocess_data(geometries, categories, tile_image_path, category, output_size=(256, 256)):
    try:
        if not os.path.exists(tile_image_path):
            logging.error(f"Image file does not exist: {tile_image_path}")
            return None, None
        
        if not validate_image(tile_image_path):
            return None, None

        with Image.open(tile_image_path) as img:
            tile_image = np.array(img)
        logging.debug(f"Loaded image with shape: {tile_image.shape}")

        x, y, zoom = parse_filename(os.path.basename(tile_image_path))
        if x is None or y is None or zoom is None:
            return None, None
        
        tile_bounds = get_tile_bounds(x, y, zoom)

        logging.debug(f"Processing tile: {tile_image_path}, bounds: {tile_bounds}")

        mask = create_mask_from_geometries(geometries, categories, tile_image.shape[:2], tile_bounds, category)

        if mask is None:
            logging.warning(f"No valid mask created for {tile_image_path} in category {category}")
            return None, None

        if np.all(mask == 0):
            logging.warning(f"Mask is all zeros for {tile_image_path} in category {category}")
            return None, None

        image_resized = tf.image.resize(tile_image, output_size)
        mask_resized = tf.image.resize(mask[..., np.newaxis], output_size)

        logging.debug(f"Preprocessed image shape: {image_resized.shape}, mask shape: {mask_resized.shape}")
        return image_resized / 255.0, mask_resized

    except Exception as e:
        logging.error(f"Error processing {tile_image_path}: {str(e)}")
        return None, None

def create_dataset(geometries, categories, tile_image_paths, category):
    images, masks = [], []

    for tile_path in tile_image_paths:
        image, mask = preprocess_data(geometries, categories, tile_path, category)
        if image is not None and mask is not None:
            images.append(image)
            masks.append(mask)

    if not images:
        logging.warning(f"No valid images found for category {category}")
        return None, None

    logging.info(f"Created dataset for category {category} with {len(images)} images.")
    return np.array(images), np.array(masks)

def check_data_distribution(geometries, categories, tile_image_paths):
    unique_categories = set(categories)
    distribution = {category: 0 for category in unique_categories}

    logging.info("Checking data distribution across categories...")
    for tile_path in tile_image_paths:
        if validate_image(tile_path):
            for category in unique_categories:
                image, mask = preprocess_data(geometries, categories, tile_path, category)
                if image is not None and mask is not None:
                    distribution[category] += 1

    for category, count in distribution.items():
        logging.info(f"Category {category}: {count} valid tiles")

    if all(count == 0 for count in distribution.values()):
        logging.error("No valid tiles found for any category. Exiting.")
        return None

    return distribution

def build_unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder (Downsampling)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bridge
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder (Upsampling)
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def train_model_for_category(geometries, categories, tile_image_paths, category, epochs=50, batch_size=32):
    logging.info(f"Training model for category: {category}")

    images, masks = create_dataset(geometries, categories, tile_image_paths, category)

    if images is None or masks is None:
        logging.warning(f"No valid data for category {category}. Skipping.")
        return None

    logging.info(f"Found {len(images)} valid images for category {category}")

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    model = build_unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {category}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {category}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    model.save(f'road_detection_model_{category}.h5')

    logging.info(f"Model training completed and saved for category: {category}")

    return model

if __name__ == "__main__":
    geojson_file = "./data/Dovellis/Dovellis TAGGED & FIXED & DISSOLVED.geojson"
    logging.info(f"Loading GeoJSON from: {geojson_file}")
    geojson_raw = load_geojson_as_raw(geojson_file)

    if geojson_raw is not None:
        geometries, categories = extract_geometries_and_categories(geojson_raw)
        logging.info(f"Extracted {len(geometries)} geometries from GeoJSON.")

        tile_directory = "./data/Dovellis/tile_data"
        logging.info(f"Searching for tile images in: {tile_directory}")
        tile_image_paths = glob(os.path.join(tile_directory, "*.png"))
        logging.info(f"Found {len(tile_image_paths)} tile images.")

        # Debug: Print some sample tile paths
        for i, path in enumerate(tile_image_paths[:5]):
            logging.info(f"Sample tile path {i}: {path}")

        data_distribution = check_data_distribution(geometries, categories, tile_image_paths)

        if data_distribution is None:
            logging.error("No valid data found for any category. Exiting.")
        else:
            unique_categories = set(categories)

            for category in unique_categories:
                if data_distribution[category] > 0:
                    train_model_for_category(geometries, categories, tile_image_paths, category)
                else:
                    logging.warning(f"Skipping category {category} due to lack of data")

            logging.info("All models trained and saved.")
    else:
        logging.error("Failed to load GeoJSON data. Exiting.")
