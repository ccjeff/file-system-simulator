#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

/*
struct node {
	int data;
	struct node* left;
	struct node* right;
};
struct node* createNode(int value) {
	struct node* newNode = (node*)malloc(sizeof(struct node));
	newNode->data = value;
	newNode->left = NULL;
	newNode->right = NULL;
	return newNode;
}
struct node* insertLeft(struct node *root, int value) {
	root->left = createNode(value);
	return root->left;
}
struct node* insertRight(struct node *root, int value) {
	root->right = createNode(value);
	return root->right;
}
*/



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
	//unsigned int count = bitLength(val);
	unsigned int count = 8;
	while (temp) {
		count -= temp & 1;
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


__device__ int countTrailingZero(int val) {
	if (val == 0) {
		return 8;
	}

	int count = 0;
	while ((val & 1) == 0)
	{
		val = val >> 1;
		count++;
	}

	//return count - 24;
	return count;
	// since the type of val is int(32 bits) and what we will store is only 8bit uchar
}


__device__ unsigned countLeadingZero(int val) {
	if (val == 0) {
		return 8;
	}
	unsigned n = 0;
	const unsigned bits = sizeof(val) * 8;
	for (int i = 1; i < bits; i++) {
		if (val < 0) break;
		n++;
		val <<= 1;
	}
	return n - 24;
}

__device__ unsigned int whereisFree(FileSystem* fs, int val) {
	// return the index of next insertion (write ptr).
	// NOTE: the ptr returned here is a superblock index rather than a direct block index. Need to be - the trailing zeros nums.
	// val: required file size
	int write_ptr = 0;
	int isCheckingFree = 0;
	int freeCounter = 0;

	// before that check for 0 in between // 1100_0011
	// call the largest zero function, usually continuous

	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		//printf("damn i am trapped\n");
		if (largestCountHelper(fs->volume[i]) > val) {
			return i;
		}
		if (freeCounter > val) {
			return write_ptr;
		}
		if (countTrailingZero(fs->volume[i]) != 0) {
			if (!isCheckingFree) {
				// this number has trailing zeros and we have not started
				// thus, start counting and move the ptr
				write_ptr = i;
				isCheckingFree = 1;
				freeCounter += countTrailingZero(fs->volume[i]);
				continue;
			}
			else {
				// is checking for the free space
				// then, further check if the number has any 1 in the number
				if (fs->volume[i] == 0) {
					// then it means the next byte taken is clean
					freeCounter += 8;
				}
				else {
					// the counting process should be terminated, since there is a 1 in the num.
					// Then, lcoate the 1. Check the number of 0s in the front and compare with the required size
					// if the fragment meets the size requirement, return the ptr. Else, move the ptr to this place and continue
					int freeSecondHalf = countLeadingZero(fs->volume[i]);
					if (freeCounter + freeSecondHalf > val) {
						return write_ptr;
					}
					else {
						// size is not large enough, continue searching, move ptr. The isCounting information does not need 
						// to be changed since the number has trailing zeros. clear free counting
						write_ptr = i;
						freeCounter = countTrailingZero(fs->volume[i]);
					}
				}
			}
		}
		else if (countLeadingZero(fs->volume[i]) == 0) {
			// there is no leading zeros
			if (isCheckingFree) {
				// check if the block number in between is larger than the size, the counting process should be finished
				isCheckingFree = 0;
				if (freeCounter > val) {
					return write_ptr;
				}
				else {
					// clear everything and start counting again
					freeCounter = 0;
				}
			}
			else {
				// then everything has not really started, nothing to worry about.
				continue;
			}
		}

	}
	return -1;
}


__device__ void changeFreeSpace(FileSystem* fs, int blockNum, int offset, int op) {
	// used to changed the free/not free information of a certain block
	if (op == 0) {
		// op == 0, the block is now changed to 'in use' state
		fs->volume[blockNum] += pow(2, 7 - offset); // TODO: bitLen - 1?
	}
	if (op == 1) {
		// op == 1, free the block in use
		fs->volume[blockNum] -= pow(2, 7 - offset);
	}

}



__device__ int findNextFree(FileSystem* fs, int size) {
	// file could occupy more than 1 block, find the first free block range.
	int free_block_counter = 0;
	int largest_free_block_counter = 0;
	int isCounting = 0;
	/*
	USING BITWISE OPERATOR INSTEAD OF CHAR TRANSLATION
	*/
	// shifting : using the original int
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		// every i is a byte
		int value = fs->volume[i];
		// count for the number of all blocks
		free_block_counter += intCountHelper(value);
	}

	if (size <= free_block_counter) {
		int byteIdx = whereisFree(fs, size);
		printf("the byteIdx info is: %d\n", byteIdx);
		printf("now the byte is: %d\n", fs->volume[byteIdx]);
		printf("trailing zero count gives: %d\n", countTrailingZero(fs->volume[byteIdx]));
		int byteOffset = 8 - countTrailingZero(fs->volume[byteIdx]);
		// return the index (first BLOCK num for data to be inserted in)
		return byteIdx * 8 + byteOffset;
	}

	return -1;
}


__device__ int findNextFCB(FileSystem* fs) {
	const int VALID_BYTE = 26;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int realAddr = fs->SUPERBLOCK_SIZE + 32 * i;
		if (fs->volume[realAddr + VALID_BYTE] == 1) {
			return i;
		}
	}
	return -1;
}



__device__ void updateFCB(FileSystem* fs, int block, int size, int time, int ptr, char *s) {
	/*
	FCB structure --- 32 byte
	filename : 20 bytes
	time: 2 bytes
	size: 2 bytes
	ptr: 2 bytes
	*/
	int address = block * 32 + fs->SUPERBLOCK_SIZE;
	for (int i = 0; i < 20; i++) {
		fs->volume[address + i] = s[i];
	}
	fs->volume[address + 20] = gtime & 0xFF;
	fs->volume[address + 21] = (gtime >> 8) & 0xFF;
	fs->volume[address + 22] = size & 0xFF;
	fs->volume[address + 23] = (size >> 8) & 0xFF;
	fs->volume[address + 24] = ptr & 0xFF;
	fs->volume[address + 25] = (ptr >> 8) & 0xFF;
	fs->volume[address + 26] = 1; // this byte is kept as a valid bit for FCB checking

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op) {
	/* Implement open operation here */
	int blockIdx = 0;
	//a. if the file exist in the file system
	/*
	1. go through all the FCBs and seek for matching names
	*/
	int filenameCounter = 0;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		for (int j = 0; j < 20; j++) {
			if (fs->volume[i * 32 + fs->SUPERBLOCK_SIZE + j] == s[j]) {
				// checking file name
				filenameCounter++;
			}
		}
		if (filenameCounter == 20) {
			return i;
			// this is a pointer to FCB 
		}
	}

	//b. if the file does not exist in the file system, create a blank file and entries/pointers, etc.
	// 1. first allocate the first free block that is available
	int fakeAllocation = findNextFree(fs, 1);
	// since it is a fake allocation (just indication where the place could possibly be since the size of the file is unknown), 
	// I didn't change the VCB here. The VCB will be changed in the write part where real allocation of storage space happens.
	if (op == 1) {
		// write mode
		// create new file
		int nextFile = findNextFCB(fs);
		// this number should be found in the directory struct (master filetable)
		updateFCB(fs, nextFile, 0, gtime, fakeAllocation, s);
		return nextFile;
	}


}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */

}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	int posFound = fp;

	int fileSize_first = fs->volume[fs->SUPERBLOCK_SIZE + posFound * 32 + 22];
	int fileSize_second = fs->volume[fs->SUPERBLOCK_SIZE + posFound * 32 + 23];
	int finalSize = (fileSize_second << 8) + fileSize_first;
	if (finalSize > size) {
		for (int i = 0; i < finalSize; i++) {
			if (i < size) {
				fs->volume[finalSize + posFound * 32 + fs->SUPERBLOCK_SIZE] = input[i];
			}
			else {
				fs->volume[finalSize + posFound * 32 + fs->SUPERBLOCK_SIZE] = 0;
			}
		}
		// then  update the size info in the FCB
		fs->volume[fs->SUPERBLOCK_SIZE + posFound * 32 + 22] = size & 0xFF;
		fs->volume[fs->SUPERBLOCK_SIZE + posFound * 32 + 23] = (size >> 8) & 0xFF;



	}





	// then perform normal writing.
	// 1. if the filesize is not enough (if the file is originally not in the fs
	// the fcb size == 1,everytime this shit should be performed

	/*REALLOCATION OF THE MOTHERFUCKING FCB*/



}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
