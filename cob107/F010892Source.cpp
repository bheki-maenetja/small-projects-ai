#include <iostream>
#include <set>
#include <queue>
#include <stack>
#include <iterator>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
using namespace std;

#define N 3 // constant to indicate the width and height of a grid

void searchAlgorithm(vector< vector<char> > s1, vector< vector<char> > s2);
set<vector< vector<char> >> dfs_search(vector< vector<char> > startState, char* file_name);

vector < vector<char> > generateNewGrid(int oldX, int oldY, int newX, int newY, vector< vector<char> > array);
vector<int> getEmptySpaceCoords(vector< vector<char> > array);
vector< vector<int> > getNeighbourCoords(int x, int y);
void writeGrid(vector< vector<char> > array, char* file_name);

bool isSolvable(string str);
string gridToString(vector< vector<char> > array);
vector< vector<char> > stringToGrid(char *str);

int main(int argc, char *argv[]) {
    // The inputs to the program are represented as 9-character strings
    if (argc != 3) {
        cout << "Invalid number of parameters\n";
        return 0;
    } else if (strlen(argv[1]) != 9 or strlen(argv[2]) != 9) {
        cout << "Invalid parameters\n";
        return 0;
    }

    vector< vector<char> > s1 = stringToGrid(argv[1]); // converts string representation of grid to an array representation
    vector< vector<char> > s2 = stringToGrid(argv[2]);

    searchAlgorithm(s1, s2);
}

void searchAlgorithm(vector< vector<char> > s1, vector< vector<char> > s2) {
    set<vector< vector<char> >> exploredS1, exploredS2;

    if (isSolvable(gridToString(s1)) == isSolvable(gridToString(s2))) { // check to see if the inputs have the same solvability
        // if s1 and s2 are both solvable or both unsolvable then the sets of their reachable
        // states are the same; therefore we only need to run DFS once

        // if both inputs have the same solvability their reachable states will be written to the same file
        exploredS1 = dfs_search(s1, "R(S1) - R(S2) - R(S1 & S2).txt");
        cout << "Number of states in R(S1): " << exploredS1.size() << "\n";
        cout << "Number of states in R(S2): " << exploredS1.size() << "\n";
        cout << "Number of states in both R(S1) and R(S2): " << exploredS1.size() << "\n";
    } else {
        // if only of s1 or s2 is solvable than the sets of their reachable states
        // will be disjoint and they will have no reachable state in common we
        // therefore need to run DFS twice

        exploredS1 = dfs_search(s1, "R(S1).txt");
        exploredS2 = dfs_search(s2, "R(S2).txt");
        cout << "Number of states in R(S1): " << exploredS1.size() << "\n";
        cout << "Number of states in R(S2): " << exploredS2.size() << "\n";
        cout << "Number of states in both R(S1) and R(S2): " << '0' << "\n";
        cout << "R(S1 & S2) = ∅" << "\n";
    }
}

set<vector< vector<char> >> dfs_search(vector< vector<char> > startState, char* file_name) {
    set<vector< vector<char> >> explored; // set to store all explored states
    stack< vector< vector<char> >> frontier; // stack data structure to to states on the edge of the search space
    frontier.push(startState);
    explored.insert(startState);

    // Iterative depth first search to find new states
    while (!frontier.empty()) {
        vector< vector<char> > grid = frontier.top();
        frontier.pop();
        vector<int> emptySpaceCoords = getEmptySpaceCoords(grid);
        vector< vector<int> > neighbours = getNeighbourCoords(emptySpaceCoords[0], emptySpaceCoords[1]);

        writeGrid(grid, file_name); // current node is written to a given file location
        for (int i = 0; i < neighbours.size(); i++) { // the loop generates all successors of the current node
            vector < vector<char> > newGrid = generateNewGrid(emptySpaceCoords[0], emptySpaceCoords[1], neighbours[i][0], neighbours[i][1], grid);
            if (explored.find(newGrid) == explored.end()) {
                explored.insert(newGrid);
                frontier.push(newGrid);
            }
        }
    }

    return explored;
}

vector< vector<int> > getNeighbourCoords(int x, int y) {
    // Returns a list of coordinates of all squares that are next to the empty square
    vector< vector<int> > neighbourCoords;
    if (x > 0)
        neighbourCoords.push_back({x-1, y});
    if (y > 0)
        neighbourCoords.push_back({x, y-1});
    if (y < N - 1)
        neighbourCoords.push_back({x, y+1});
    if (x < N - 1)
        neighbourCoords.push_back({x+1, y});
    
    return neighbourCoords;
}

vector < vector<char> > generateNewGrid(int oldX, int oldY, int newX, int newY, vector< vector<char> > array) {
    // Generates a new grid by placing the empty square at new coordinates
    // The value at the new coordinates is then placed in previous location of the empty square
    vector < vector<char> > newGrid;
    copy(array.begin(), array.end(), back_inserter(newGrid));

    char temp = newGrid[newX][newY];
    newGrid[newX][newY] = '0';
    newGrid[oldX][oldY] = temp;
    
    return newGrid;
}

vector<int> getEmptySpaceCoords(vector< vector<char> > array) {
    // Returns a coordinate pair representing the location of the empty square
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            if (array[i][j] == '0') return {i,j};
    }
}

string gridToString(vector< vector<char> > array) {
    // Converts a 2d array representation of a grid to a 9 character string representation
    string newString;
    newString.resize(9);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            newString[j + N*i] = array[i][j];
    }

    return newString;
}

vector< vector<char> > stringToGrid(char *str) {
    // Converts a string representation of a grid to a 2d array representation
    vector< vector<char> > newGrid = {{'-', '-', '-'}, {'-', '-', '-'}, {'-', '-', '-'}};
    for (int i = 0; i < 9; i++)
        newGrid[i / 3][i%3] = str[i];
    return newGrid;
}

bool isSolvable(string str) {
    // Checks the solvability of a grid
    // The grid is input as a string representation
    // Solvability is checked by counting the number of inversions:
    // if the number of inversions is even the grid is solvable; otherwise it is unsolvable
    int inv_count = 0;
    for (int i = 0; i < N*N - 1; i++) {
        for (int j = i+1; j < N*N; j++)
            if (str[j] != '0' && str[i] != '0' &&  str[i] > str[j])
                inv_count++;
    }

    return inv_count % 2 == 0;
}

void writeGrid(vector< vector<char> > array, char* file_name) {
    // Writes a grid to a given file
    // This is how the program will output R(S1) and R(S2)
    string filename(file_name);
    ofstream MyFile;

    MyFile.open(filename, std::ios_base::app);
    if (array.size() == N)
        for (int i = 0; i < N; i++)
        {
            MyFile << array[i][0] << " " << array[i][1] << " " << array[i][2] << "\n";
        }
    MyFile << "\n";
    MyFile.close();
}