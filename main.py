import math
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import random

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
          
          int idx = threadIdx.x + threadIdx.y * width;

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
          
          if(threadIdx.x < (width - 1))
            atomicAdd(&energy, abs(image_device[red_idx] - image_device[red_idx + 3])
                             + abs(image_device[green_idx] - image_device[green_idx + 3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + 3]));
          
          if(threadIdx.y < (width - 1))
            atomicAdd(&energy, abs(image_device[red_idx] - image_device[red_idx + width*3])
                             + abs(image_device[green_idx] - image_device[green_idx + width*3])
                             + abs(image_device[blue_idx] - image_device[blue_idx + width*3]));
                             
          __syncthreads();

          if (threadIdx.x == 0 && threadIdx.y == 0)
              *result = energy;
      }
      
      __global__ void simulated_annealing(unsigned char *image_device, int width, int *result, int *to_swap, int *results_host, float temp, float p){
        //extern __shared__ image[32 * 32 * 3];
        
        __shared__ int results[8];
        if(threadIdx.x == 0)
            results[threadIdx.y] = 0;
            
        //pixel je nasumican piksel, a pixel2 je odabrani sused
        int y = to_swap[threadIdx.y*3];
        int x = to_swap[threadIdx.y*3+1];
        int pixel = x*3 + y * 3*width;
        int pixel2 = (to_swap[threadIdx.y*3+2] == 1) ? pixel + 3 : pixel + 3*width;
        
        int idx = pixel;
        
        //odredjujemo za koji piksel ce nit da racuna energiju
        switch(threadIdx.x){
          case 0: x -= 1; y -= 1; break;
          case 1: y -= 1; break;
          case 2: x += 1; y -= 1; break;
          case 3: x -= 1; break;
          case 4: break;
          case 5: x += 1; break;
          case 6: x -= 1; y += 1; break;
          case 7: y += 1; break;
          case 8: x += 1;y += 1; break;
          case 9: x += (to_swap[threadIdx.y*3+2] == 1) ? 2 : -1; y += (to_swap[threadIdx.y*3+2] == 1) ? -1 : 2; break;
          case 10: x += (to_swap[threadIdx.y*3+2] == 1) ? 2 : 0; y += (to_swap[threadIdx.y*3+2] == 1) ? 0 : 2; break;
          case 11: x += (to_swap[threadIdx.y*3+2] == 1) ? 2 : 1; y += (to_swap[threadIdx.y*3+2] == 1) ? 1 : 2; break;
        }
        
        //idx je kordinata piksela koji nit treba da obradi
        idx = x*3 + y * 3*width;
        
        //offsetovi za odredjene boje
        
        int red_idx = idx;
        int green_idx = red_idx+1;
        int blue_idx = red_idx+2;
        int cell_energy_before = 0, cell_energy_after = 0;
        
        //prvo proveramavo da li je idx validan
        if(x > -1 && x < width && y > -1 && y < width){
        
          //racunamo energiju sa desnim ako postoji
          if(x < width-1 || (idx == pixel2 && (to_swap[threadIdx.y*3+2] == 1))){
            
            //pre promene
            cell_energy_before += abs(image_device[red_idx] - image_device[red_idx + 3])
                                + abs(image_device[green_idx] - image_device[green_idx + 3])
                                + abs(image_device[blue_idx] - image_device[blue_idx + 3]);
            
            //posle promene
            if(idx == pixel && ((idx + 3) == pixel2)){
                cell_energy_after += abs(image_device[pixel2 + 3] - image_device[pixel])
                                   + abs(image_device[pixel2 + 1 + 3] - image_device[pixel+1])
                                   + abs(image_device[pixel2 + 2 + 3] - image_device[pixel+2]);
            }
            else if(idx == pixel2 && ((idx - 3) == pixel)){
            
                cell_energy_after += abs(image_device[pixel2] - image_device[pixel])
                                   + abs(image_device[pixel2 + 1] - image_device[pixel+1])
                                   + abs(image_device[pixel2 + 2] - image_device[pixel+2]);
            }
            else if(idx == pixel){
                cell_energy_after += abs(image_device[pixel] - image_device[pixel2 + 3])
                                   + abs(image_device[pixel + 1] - image_device[(pixel2+1) + 3])
                                   + abs(image_device[pixel + 2] - image_device[(pixel2+2) + 3]);
            }
            else if(idx == pixel2){
                cell_energy_after += abs(image_device[pixel2] - image_device[pixel + 3])
                                   + abs(image_device[pixel2 + 1] - image_device[(pixel+1) + 3])
                                   + abs(image_device[pixel2 + 2] - image_device[(pixel+2) + 3]);
            }
            else if(idx+3 == pixel){
                cell_energy_after += abs(image_device[red_idx] - image_device[pixel2])
                                   + abs(image_device[green_idx] - image_device[(pixel2+1)])
                                   + abs(image_device[blue_idx] - image_device[(pixel2+2)]);
            }
            else if(idx+3 == pixel2){
                cell_energy_after += abs(image_device[red_idx] - image_device[pixel])
                                   + abs(image_device[green_idx] - image_device[(pixel+1)])
                                   + abs(image_device[blue_idx] - image_device[(pixel+2)]);
            }
            else{
                cell_energy_after += abs(image_device[red_idx] - image_device[red_idx + 3])
                                   + abs(image_device[green_idx] - image_device[green_idx + 3])
                                   + abs(image_device[blue_idx] - image_device[blue_idx + 3]);
            }
          }
          
          //racunamo energiju sa donjim ako postoji
          if(y < width - 1 || (idx == pixel2 && (to_swap[threadIdx.y*3+2] == 0))){

            //pre promene
            cell_energy_before += abs(image_device[red_idx] - image_device[red_idx + width*3])
                                + abs(image_device[green_idx] - image_device[green_idx + width*3])
                                + abs(image_device[blue_idx] - image_device[blue_idx + width*3]);
            
            //posle promene
            if(idx == pixel && (idx + 3*width) == pixel2){
                cell_energy_after += abs(image_device[pixel2 + 3*width] - image_device[pixel])
                                   + abs(image_device[(pixel2 + 1) + 3*width] - image_device[pixel+1])
                                   + abs(image_device[(pixel2 + 2) + 3*width] - image_device[pixel+2]);
            }
            else if(idx == pixel2 && (idx - 3*width) == pixel){
                cell_energy_after += abs(image_device[pixel2] - image_device[pixel])
                                   + abs(image_device[pixel2 + 1] - image_device[pixel+1])
                                   + abs(image_device[pixel2 + 2] - image_device[pixel+2]);
            }
            else if(idx == pixel){
                cell_energy_after += abs(image_device[pixel] - image_device[pixel2 + width*3])
                                   + abs(image_device[pixel + 1] - image_device[(pixel2+1) + width*3])
                                   + abs(image_device[pixel + 2] - image_device[(pixel2+2) + width*3]);
            }
            else if(idx == pixel2){
                cell_energy_after += abs(image_device[pixel2] - image_device[pixel + width*3])
                                   + abs(image_device[pixel2 + 1] - image_device[(pixel+1) + width*3])
                                   + abs(image_device[pixel2 + 2] - image_device[(pixel+2) + width*3]);
            }
            else if(idx+width*3 == pixel){
                cell_energy_after += abs(image_device[red_idx] - image_device[pixel2])
                                   + abs(image_device[green_idx] - image_device[(pixel2+1)])
                                   + abs(image_device[blue_idx] - image_device[(pixel2+2)]);
                
            }
            else if(idx+width*3 == pixel2){
                cell_energy_after += abs(image_device[red_idx] - image_device[pixel])
                                   + abs(image_device[green_idx] - image_device[pixel+1])
                                   + abs(image_device[blue_idx] - image_device[pixel+2]);
            }
            else{
                cell_energy_after += abs(image_device[red_idx] - image_device[red_idx + width*3])
                                   + abs(image_device[green_idx] - image_device[green_idx + width*3])
                                   + abs(image_device[blue_idx] - image_device[blue_idx + width*3]);
            }
          }
        }
        atomicAdd(&results[threadIdx.y], cell_energy_after - cell_energy_before);
        
        __syncthreads();
        
        if(threadIdx.x == 0)
            results_host[threadIdx.y] = results[threadIdx.y];
        
        __syncthreads();
        
        if(threadIdx.x == 0 && threadIdx.y == 0){
            float de = results[0];
            int j;
            for(int i = 1; i < 8; i++){
                if(de > results[i]){
                    de = results[i];
                    j = i;
                }
            }
            printf("najbolja promena: %0.0f => [%d, %d] <-> [%d, %d]", de, to_swap[j*3], to_swap[j*3+1],
                                                                  (to_swap[j*3+2]) ? to_swap[j*3]: to_swap[j*3]+1,
                                                                  (to_swap[j*3+2]) ? to_swap[j*3+1]+1 : to_swap[j*3+1]);
            if(de < 0 || p < (-de / temp)*(-de / temp)){
                //prihvaceno
            }
        }
      }
   """)
if __name__ == "__main__":
    width = 5
    #image = np.random.randint(0, 2, (5, 5, 3), dtype=np.uint8)#np.random.randint(0, 255, (width, width, 3), dtype=np.uint8)
    image = np.array([[[0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]],
                      [[1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                      [[1, 0, 1], [0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
                      [[0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]], dtype=np.uint8)

    """DEFINISANJE PARAMETARE"""
    image_gpu = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(image_gpu, image)

    energy = np.zeros(1, dtype=np.int32)
    energy_gpu = cuda.mem_alloc(energy.nbytes)
    cuda.memcpy_htod(energy_gpu, energy)

    results = np.zeros(8, dtype=np.int32)
    results_gpu = cuda.mem_alloc(results.nbytes)
    cuda.memcpy_htod(results_gpu, results)

    to_swap = [[2, 0, 1], [2, 0, 0], [0, 0, 0], [0, 3, 1], [0, 2, 1], [2, 3, 0], [3, 1, 0], [0, 2, 0]]  # generate_to_swap()
    to_swap_gpu = cuda.mem_alloc(np.array(to_swap).nbytes)
    cuda.memcpy_htod(to_swap_gpu, np.array(to_swap))

    start_temp = np.array([100], dtype=np.float32)
    temp_gpu = cuda.mem_alloc(start_temp.nbytes)
    cuda.memcpy_htod(temp_gpu, start_temp)

    probability = np.array([random.random()], dtype=np.float32)
    probability_gpu = cuda.mem_alloc(probability.nbytes)
    cuda.memcpy_htod(probability_gpu, probability)

    """POZIV KERNELA"""
    calculate_energy = mod.get_function("calculate_energy")  # 32, 32, 1        grid=(math.ceil(width/32), math.ceil(width/32), 1))
    calculate_energy(image_gpu, energy_gpu, np.int32(width), block=(5, 5, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(energy, energy_gpu)
    cuda.memcpy_dtoh(image, image_gpu)
    print("energija: ", energy[0])

    simulated_annealing = mod.get_function("simulated_annealing")
    simulated_annealing(image_gpu, np.int32(width), energy_gpu, to_swap_gpu, results_gpu, temp_gpu, probability_gpu, block=(12, 8, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(results, results_gpu)
    print("\nresultati 8 swapova: ", results)

    # starting_temp = 100
    # total = 30_000_000
    # for iteration in tqdm(range(total)):
    #     t = iteration / total
    #     temp = (1 - t) * starting_temp
    #     image2[src[0], src[1]] = image[tgt[0], tgt[1]]
    #     image2[tgt[0], tgt[1]] = image[src[0], src[1]]
    #
    #     dE = energy_delta(image, image2, src, tgt)
    #     if dE < 0 or random.random() < np.exp2(-dE / temp):
    #         image[src[0], src[1]] = image2[src[0], src[1]]
    #         image[tgt[0], tgt[1]] = image2[tgt[0], tgt[1]]
    #         # current_energy = new_energy
    #         current_energy += dE
    #         swaps += 1
    #     else:
    #         image2[src[0], src[1]] = image[src[0], src[1]]
    #         image2[tgt[0], tgt[1]] = image[tgt[0], tgt[1]]
