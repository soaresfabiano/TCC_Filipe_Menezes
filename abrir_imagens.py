import os
import nibabel as nib
import cv2
import numpy as np

def print_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        print(f"Arquivo encontrado: {filename}")

def load_and_display_nii_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"): 
            print(f"Lendo o arquivo: {filename}")  # Imprime o nome do arquivo
            img = nib.load(os.path.join(folder_path, filename))
            img_data = img.get_fdata()

            print(f"Forma dos dados da imagem: {img_data.shape}")  # Verifica a forma dos dados da imagem

            # Normaliza a imagem para o intervalo [0, 255] para exibição correta
            img_data_normalized = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Exibe a imagem normalizada usando cv2
            cv2.imshow(filename + " (normalizada)", img_data_normalized[:, :, img_data_normalized.shape[2] // 2])

            # Exibe a imagem original usando cv2
            cv2.imshow(filename + " (original)", img_data[:, :, img_data.shape[2] // 2])
            
    # Aguarda até que todas as janelas de figura sejam fechadas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Carrega o caminho da pasta 
folder_path = 'c:\\Users\\Z004N5ZE\\Documents\\2023.2\\tcc1\\datasets\\Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information\\Patient-7'
print_all_files_in_folder(folder_path)
load_and_display_nii_images(folder_path)
