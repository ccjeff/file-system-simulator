#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}

__device__ int pow(int base, int power) {
	int result = 1;
	while (power != 0) {
		result *= base;
		power--;
	}
	return result;
}

__device__ int bitLength(int bin) {
	int counter = 1;
	int result = bin;
	while (result / 2 != 0) {
		counter++;
		result = result / 2;
	}
	return counter;
}



__device__ int convertBinaryToDecimal(u32 n) {
	int decimalNumber = 0, i = 0, remainder;
	while (n != 0)
	{
		remainder = n % 10;
		n /= 10;
		decimalNumber += remainder * pow(2, i);
		++i;
	}
	return decimalNumber;
}

__device__ u32 convertDecimalToBinary(int n) {
	u32 binaryNumber = 0;
	int remainder, i = 1, step = 1;
	while (n != 0)
	{
		remainder = n % 2;
		n /= 2;
		binaryNumber += remainder * i;
		i *= 10;
	}
	return binaryNumber;
}


__device__ unsigned int intCountHelper(int val) {
	// count the number of 0 (instead of 1 since 1 is set state)
	unsigned int temp = val;
	unsigned int count = bitLength(val);
	while (temp) {
		count -= temp&1;
		temp >>= 1;
	}
	return count;
}

/*
this function is returning the number of fragments within the 8 block set, yet the location of the fragment is not located
*/
__device__ unsigned int largestCountHelper(int val) {
	unsigned int temp = 255 - val;
	// since counting can be done easier with set bit at 1, however 0 is needed
	unsigned int count = 0;
	while (temp != 0) {
		temp = temp & (temp << 1);
		count++;
	}
	return count;
}


__device__ unsigned int whereisLargest(int val) {
	// TODO: return the index of the largest fraction of 0s.
}


__device__ void changeFreeSpace(FileSystem* fs, int blockNum, int offset, int op) {
	// used to changed the free/not free information of a certain block
	if (op == 0) {
		// op == 0, the block is now changed to 'in use' state
		int bitLen = bitLength(fs->volume[blockNum]);
		fs->volume[blockNum] += pow(2, bitLen - offset); // TODO: bitLen - 1?

		/*debug section*/
		//printf("the result after change: %d\n", fs->volume[blockNum]);
		
	}
	if (op == 1) {
		// op == 1, free the block in use
		int bitLen = bitLength(fs->volume[blockNum]);
		fs->volume[blockNum] -= pow(2, bitLen-1-offset);
	}
}

__device__ int findNextFree(FileSystem* fs, int size) {
	// file could occupy more than 1 block, find the first free block range.
	int free_block_counter = 0;
	int largest_free_block_counter = 0;
	int isCounting = 0;
	/*
	TODO: testing between bitwise operations and direct char* transformations
	*/
	// shifting : using the original int
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		// every i is a byte
		int value = fs->volume[i];
		free_block_counter += intCountHelper(value);
		if (largest_free_block_counter < largestCountHelper(value)) {
			// largest_free_block_counter = largestCountHelper(value);
		}
	}

	return -1;
}




__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	int blockIdx = 0;
	//a. if the file exist in the file system
	/*
	1. 
	*/
	//b. if the file does not exist in the file system, create a blank file and entries/pointers, etc.




	/*tester codes*/
	
	int number = 255;
	int next = 0;
	int nexter = 0;
	next = convertDecimalToBinary(number);
	nexter = convertBinaryToDecimal(next);
	printf("number in binary is: %d \n",next);
	printf("number in uchar is: %d \n", (uchar)number);
	printf("bit length of number is: %d \n", bitLength(number));
	printf("number in decimal is: %d\n", nexter);
	printf("global counter gives: %d\n", intCountHelper(200));
	printf(">>>>>>largest iteration test>>>>>\n");
	printf("the result of the largestCount: %d\n", largestCountHelper(200));
	printf("200: %d\n", convertDecimalToBinary(200));
	fs->volume[0] = (uchar)number;
	changeFreeSpace(fs,0,2,1);
	printf("number after free space changer: %d\n", fs->volume[0]);
	printf("with the binary representation : %d \n", convertDecimalToBinary(fs->volume[0]));
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
