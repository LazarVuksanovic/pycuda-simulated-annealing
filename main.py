import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


def energy(image):
    distances = np.zeros_like(image, dtype=np.uint8)
    distances[:, :-1, :] += np.abs(image[:, :-1, :] - image[:, 1:, :])
    distances[:-1, :, :] += np.abs(image[:-1, :, :] - image[1:, :, :])
    return distances.sum()


# def energy_delta(image, image2, src, tgt):
#   row_low = max(0, src[0] - 1)
#   row_high = tgt[0] + 2
#   col_low = max(0, src[1] - 1)
#   col_high = tgt[1] + 2
#   old = calculate_energy(image[row_low:row_high, col_low:col_high], result_gpu, np.uint8(width), np.uint8(width), block=(12, 1, 1), grid=(1, 1, 1))
#   new = calculate_energy(image2[row_low:row_high, col_low:col_high], result_gpu, np.uint8(width), np.uint8(width), block=(12, 1, 1), grid=(1, 1, 1))
#   return new - old

def generate_to_swap():
    global image
    m = []
    for i in range(0,8):
        src = list(np.random.randint(0, image.shape[0] - 1, 2))
        src.append(np.random.randint(0, 2))
        m.append(src)
    return m

mod = SourceModule("""
      __global__ void calculate_energy(unsigned char *image, int *result, int width) {
          __shared__ int energy;
          __shared__ unsigned char image_device[32*32*3];
          
          //int idx = threadIdx.x + blockDim.x * blockIdx.x + threadIdx.y * width + blockDim.y * blockIdx.y * width * 3;
          int idx = threadIdx.x + threadIdx.y * width;

          int x = threadIdx.x;
          int y = threadIdx.y;

          int red_idx = idx*3;
          int green_idx = red_idx+1;
          int blue_idx = red_idx+2;
          //int local_energy_red, local_energy_green, local_energy_blue;

          image_device[red_idx] = image[red_idx];
          image_device[green_idx] = image[green_idx];
          image_device[blue_idx] = image[blue_idx];

          if (threadIdx.x == 0 && threadIdx.y == 0)
              energy = 0;

          __syncthreads();
          
          if(x < width - 1)
            atomicAdd(&energy, abs(image_device[red_idx] - image_device[red_idx + 3])
                             + abs(image_device[green_idx] - image_device[green_idx + 3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + 3]));
          
          if(y < width - 1)
            atomicAdd(&energy, abs(image_device[red_idx] - image_device[red_idx + width*3])
                             + abs(image_device[green_idx] - image_device[green_idx + width*3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + 3]));
          
          __syncthreads();

          if (threadIdx.x == 0 && threadIdx.y == 0)
              *result = energy;
      }
      
      __global__ void simulated_annealing(int *image_host, int width, int *result, int *to_swap){
        extern __shared__ image[32 * 32 * 3];

        //pixel je nasumican piksel, a pixel2 je odabrani sused
        int pixel = to_swap[threadIdx.y][0] + to_swap[threadIdx.y][1] * width;
        int pixel2 = (to_swap[threadIdx.y][2] == 1) ? pixel + 1 : pixel + width;

        int idx = pixel;
        int idx2 = pixel2
        
        //odredjujemo za koji piksel ce nit da racuna energiju
        switch(threadIdx.x % 8){
          case 0: idx -= width + 1;
          case 1: idx -= width; break;
          case 2: idx -= width + 1; break;
          case 3: idx -= 1; break;
          case 4: idx = idx; break;
          case 5: idx += 1; break;
          case 6: idx += width - 1; break;
          case 7: idx += width; break;
          case 8: idx += width + 1; break;
          case 9: idx = (to_swap[threadIdx.y][2] == 1) ? (idx2 - width) + 1 : (idx2 + width) -1; break;
          case 10: idx = (to_swap[threadIdx.y][2] == 1) ? idx2 + 1 : idx2 + width; break;
          case 11: idx = idx2 + width+1; break;
        }
        
        //offsetovi za odredjene boje
        int red_idx = idx*3;
        int green_idx = red_idx+1;
        int blue_idx = red_idx+2;
        int cell_energy_before = 0, cell_energy_after = 0;
        
        int desno, levo;
        
        //prvo proveramavo da li je idx validan
        if(idx >= 0 && idx < width){
        
          //racunamo energiju pre promene, (dodajemo ako suseda ima)
          if((idx % width) < width - 1){
            desno = abs(image_device[red_idx] - image_device[red_idx + 3])
                             + abs(image_device[green_idx] - image_device[green_idx + 3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + 3]);
            atomicAdd(&cell_energy_before, desno);
          }
          
          if((idx / width) < width - 1)
            levo = abs(image_device[red_idx] - image_device[red_idx + width*3])
                             + abs(image_device[green_idx] - image_device[green_idx + width*3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + 3]);
            atomicAdd(&cell_energy_before, levo);
            
          //racunamo energiju posle promene
          //if((idx % width) < width - 1){
            
          //}
        }

        if(to_swap[threadIdx.y][2] == 1){
            
        }
        else{
          //menjamo sa donjim
        }
      }
   """)
if __name__ == "__main__":
    width = 32
    image = np.random.randint(0, 255, (width, width, 3), dtype=np.uint8)#np.random.randint(0, 255, (width, width, 3), dtype=np.uint8)
    print(image)
    # definisanje cuda parametara
    image_gpu = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(image_gpu, image)

    result = np.zeros(1, dtype=np.int32)
    result_gpu = cuda.mem_alloc(result.nbytes)
    cuda.memcpy_htod(result_gpu, result)

    # poziv kernela
    calculate_energy = mod.get_function("calculate_energy")  # 32, 32, 1        grid=(math.ceil(width/32), math.ceil(width/32), 1))
    calculate_energy(image_gpu, result_gpu, np.int32(width), block=(32, 32, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(result, result_gpu)
    cuda.memcpy_dtoh(image, image_gpu)
    # print("posle\n", image)
    print("rezultat: ", result[0])
    print("provera: ", energy(image))

    # suma = np.zeros(1, dtype=np.int32)
    # suma_gpu = cuda.mem_alloc(suma.nbytes)
    # cuda.memcpy_htod(suma_gpu, suma)

    simulated_annealing = mod.get_function("simulated_annealing")

    simulated_annealing(image_gpu, np.int32(width), result_gpu, generate_to_swap(), block=(12, 8, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(suma, suma_gpu)
    print(suma)



# ponavljamo zamenu 100 puta, to_swap ima matrice za obradu zamene
# simulated_annealing = mod.get_function("simulated_annealing")
# moves = np.array([(0, 1), (1, 0)], dtype=np.int32)
# image2 = image
# for i in range(1,100):
#   src = np.random.randint(0, image.shape[0]-1, 2)
#   move = moves[np.random.randint(0, 2)]
#   tgt = src + move
#   image2[src[0], src[1]] = image[tgt[0], tgt[1]]
#   image2[tgt[0], tgt[1]] = image[src[0], src[1]]
#   dE = energy_delta(image, image2, src, tgt)
#   simulated_annealing(image_gpu, np.int32(width), result_gpu, to_swap, block(12, 8, 1), grid(1,1,1))