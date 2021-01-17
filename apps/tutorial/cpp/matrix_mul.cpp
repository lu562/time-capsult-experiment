#include <NTL/ZZ_pXFactoring.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>


#define LOCK_FILE_NAME "lock.file"


using namespace std;
using namespace NTL;
using namespace std::chrono;

inline ZZ readZZ(istream& stream)
{
    string temp;
    getline (stream, temp);
    temp += "\0";
    return conv<ZZ>(temp.c_str());
}

inline ZZ_p readZZ_p(istream& stream)
{
    string temp;
    getline (stream, temp);
    temp += "\0";
    return to_ZZ_p(conv<ZZ>(temp.c_str()));
}

inline int readInt(istream& stream)
{
    string temp;
    getline (stream, temp);
    temp += "\0";
    return atoi(temp.c_str());
}

void writeToFile(
        const string& outputFileName,
        Mat<ZZ_p> res,
        const ZZ& fieldModulus
    )
{
    cout << "here?aaa" << endl;
    fstream outputFile(outputFileName, ios::out);
    long row = res.NumRows();
    long column = res.NumCols();
    cout << (int)row << endl;
    cout << (int)column << endl;
    outputFile << fieldModulus << endl;
    outputFile << (int)row << endl;
    outputFile << (int)column << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            outputFile << res[i][j] << endl;
            // cout << "value: " << input_a[i][j] << endl;
        }
    }

    // Close the file.
    outputFile.close();
}

/**
 * This method uses a lock file. It takes a lock on the lock file and then
 * writes on the original file. Locking will not work if the original file is
 * modified by some other process which doesn't take a lock on the same lock
 * file.
 * */
void writeUsingLockFile(
        const string& outputFileName,
        Mat<ZZ_p> res,
        const ZZ& fieldModulus
    )
{
    //Declare and set up a flock object from fcntl.
    struct flock outputFileLock;
    outputFileLock.l_type = F_WRLCK; /* Write lock */
    outputFileLock.l_whence = SEEK_SET;
    outputFileLock.l_start = 0;
    outputFileLock.l_len = 0; /* Lock whole file */

    // Open lock file in write mode and request lock.
    FILE* outputFile = fopen(LOCK_FILE_NAME, "w");
    if (outputFile == NULL)
    {
        cout << "Something bad happened\n";
        exit(1);
    }

    // Get file descriptor associated with file handle.
    int fd = fileno(outputFile);

    // Request the lock.  We will wait forever until we get the lock.
    if (fcntl(fd, F_SETLKW, &outputFileLock) == -1)
	{
        cout << "ERROR: could not obtain lock on " << outputFile << '\n';
        exit(1);
	}

    writeToFile(outputFileName, res, fieldModulus);

    // Release lock and close file.
    outputFileLock.l_type = F_UNLCK;
    if (fcntl(fd, F_UNLCK, &outputFileLock) == -1)
	{
	    cout << "ERROR: could not release lock on " << outputFile << '\n';
        exit(1);
 	}

    fclose(outputFile);
}

/**
 * Input:
 * fieldModulus: Modulus of the field.
 * a : Number whose powers need to be computed.
 * a-b : Opened value of a - b
 * k : Number of powers to be computed.
 * bs : k Pre computed powers of some random number.
 * */
void runWithInputs(const string& inputFileName1, const string& inputFileName2, const string& outputFileName)
{
    ifstream inputFile(inputFileName1);

    ZZ fieldModulus = readZZ(inputFile);
    cout << "modulus: " << fieldModulus << endl;

    //Initialize the field with the modulus.
    ZZ_p::init(ZZ(fieldModulus));
    unsigned int row = readInt(inputFile);
    unsigned int column = readInt(inputFile);
    Mat<ZZ_p> input_a;
    input_a.SetDims(row, column);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            input_a[i][j] = readZZ_p(inputFile);
            // cout << "value: " << input_a[i][j] << endl;
        }
    }
    inputFile.close();
    ifstream inputFile2(inputFileName2);
    fieldModulus = readZZ(inputFile2);

    // Initialize the field with the modulus.
    row = readInt(inputFile2);
    column = readInt(inputFile2);
    Mat<ZZ_p> input_b;
    input_b.SetDims(row, column);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            input_b[i][j] = readZZ_p(inputFile2);
            // cout << "value: " << input_a[i][j] << endl;
        }
    }
    inputFile2.close();

    Mat<ZZ_p> result;
    mul(result, input_a, input_b);
    writeUsingLockFile(outputFileName, result, fieldModulus);
}

int main(int argc, char* argv[])
{
    runWithInputs(string(argv[1]), string(argv[2]), string(argv[3]));
    return 0;
}