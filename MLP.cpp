#include <bits/stdc++.h>
using namespace std;

#define eps 0.1
#define db if(0) cout
typedef unsigned int uint;

vector<double> getInput(const string &input){
    istringstream line(input);
    vector<double> v;
    double d;
    while(line>>d) v.push_back(d);
    return v;
}

void printVector(vector<double> &v, bool teste = false){
    for(int i = 0, len = v.size(); i < len; i++){
        if(teste) cout<<(v[i] < eps ? 0 : v[i])<<" \n"[i == len - 1];
        else db<<(v[i] < eps ? 0 : v[i])<<" \n"[i == len - 1];
    }
}

struct Aresta{
    double peso;
    double deltaPeso;
};

struct No;
typedef vector<No> Layer;

struct No{
    uint index;
    double valor, m_gradient;
    vector<Aresta> arestas;
    static double eta;      ///overall net learning rate [0 : 1]
    static double alpha;    ///momentum[0 : n]

    static double ativacao(double x){return tanh(x);}   ///hyperbolic tangent [-1 : 1 ]

    static double derivadaAtivacao(double x){return 1.0 - x * x;}

    No(uint numOutputs, uint myIndex){
        for(uint c = 0; c < numOutputs; c++){
            arestas.push_back(Aresta());
            arestas.back().peso = rand()/double(RAND_MAX);
        }
        index = myIndex;
    }

    void setOutputVal(double val){valor = val;}

    double getOutputVal() const{return valor;}

    void calcOutputGradients(double targetVal){
        double delta = targetVal - valor;
        m_gradient = delta * No::derivadaAtivacao(valor);
    }

    double sumDOW(const Layer &nextLayer){
        double sum = 0.0;
        for(uint n = 0; n < nextLayer.size() - 1; n++){
            sum += arestas[n].peso * nextLayer[n].m_gradient;
        }
        return sum;
    }

    void calcHiddenGradients(const Layer &nextLayer){
        double dow = sumDOW(nextLayer);
        m_gradient = dow * No::derivadaAtivacao(valor);
    }

    void atualizarPesos(Layer &prevLayer){
        for(uint n = 0; n < prevLayer.size(); n++){
            No &no = prevLayer[n];
            double oldDeltaPeso = no.arestas[index].deltaPeso;

            double newDeltaPeso = (eta * no.valor * m_gradient) + (alpha * oldDeltaPeso);

            no.arestas[index].deltaPeso = newDeltaPeso;
            no.arestas[index].peso += newDeltaPeso;
        }
    }

    void feedForward(const Layer &prevLayer){
        double sum = 0.0;

        for(uint n = 0; n < prevLayer.size(); n++){
            sum += prevLayer[n].valor *
                   prevLayer[n].arestas[index].peso;
        }
        valor = No::ativacao(sum);
    }
};

double No::eta = 0.15;
double No::alpha = 0.5;

struct MLP{

    vector<Layer> layers;
    double erro, erroMedio, erroSmoothingFactor;

    MLP(){}

    MLP(const vector<uint> &topology){
        uint numLayers = topology.size();
        for(uint layerNum = 0; layerNum < numLayers; layerNum++){
            layers.push_back(Layer());
            uint numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
            for(uint noNum = 0; noNum <= topology[layerNum]; noNum++){
                layers.back().push_back(No(numOutputs, noNum));
            }
        }
        layers.back().back().valor = 1.0;
    }

    void feedForward(const vector<double> &input){
        assert(input.size() == layers[0].size() - 1);

        for(uint i = 0; i < input.size(); i++){
            layers[0][i].valor = input[i];
        }

        for(uint layerNum = 1; layerNum < layers.size(); layerNum++){
            Layer &prevLayer = layers[layerNum - 1];
            for(uint i = 0; i < layers[layerNum].size() - 1; i++){
                layers[layerNum][i].feedForward(prevLayer);
            }
        }
    }

    void backProp(const vector<double> &target){
        Layer &outputLayer = layers.back();
        erro = 0.0;

        for(uint i = 0, len = i < outputLayer.size() - 1; i < len; i++){
            double delta = target[i] - outputLayer[i].getOutputVal();
            erro += delta * delta;
        }

        erro /= outputLayer.size() - 1;
        erro = sqrt(erro);

        erroMedio = (erroMedio * erroSmoothingFactor + erro)/(erroSmoothingFactor + 1.0);

        for(uint n = 0, len = outputLayer.size() - 1; n < len; n++){
            outputLayer[n].calcOutputGradients(target[n]);
        }

        for(uint layerNum = layers.size() - 2; layerNum > 0; layerNum--){
            Layer &hiddenLayer = layers[layerNum];
            Layer &nextLayer = layers[layerNum + 1];
            for(uint n = 0; n < hiddenLayer.size(); n++){
                hiddenLayer[n].calcHiddenGradients(nextLayer);
            }
        }

        for(uint layerNum = layers.size() - 1; layerNum > 0; layerNum--){
            Layer &layer = layers[layerNum];
            Layer &prevLayer = layers[layerNum - 1];
            for(uint n = 0; n < layer.size() - 1; n++){
                layer[n].atualizarPesos(prevLayer);
            }
        }
    }

    void getResults(vector<double> &result) const {
        result.clear();
        for(uint n = 0; n < layers.back().size() - 1; n++){       ///-1 por causa do bias
            result.push_back(layers.back()[n].valor);
        }
    }
};

uint hiddenSize(int inputSize = 0, int outputSize = 0, int trainingSize = 0){
    if(trainingSize)return trainingSize / ((rand() % 6 + 5) * (inputSize + outputSize));
    if(outputSize) return (inputSize + outputSize + 1)/2;
    return 1 + (rand() % (inputSize + 1));
}

MLP mlp;

void trainMLP(string inputFile, string targetFile){
    ///define topologia da MLP
    vector<uint> topology = {2, hiddenSize(2), 1};
    mlp = MLP(topology);

    ///le arquivos
    ifstream input (inputFile);
    if (!input.is_open()) {cout<<"treino: input deu problema\n"; return;}

    ifstream target (targetFile);
    if (!target.is_open()) {cout<<"treino: target deu problema\n"; return;}

    ///inicia treinamento
    vector<double> inputData;
    vector<double> targetData;
    vector<double> result;

    string line;
    while(getline(input, line)){

        ///feed forward
        inputData = getInput(line);
        mlp.feedForward(inputData);
        db<<"Entrada: "; printVector(inputData);

        ///obtem resultado atual da rede
        mlp.getResults(result);
        db<<"Saida  : "; printVector(result);

        ///back propagation
        getline(target, line);
        targetData = getInput(line);
        db<<"Target : "; printVector(targetData);
        mlp.backProp(targetData);

        ///mostra erro medio
        db<<"Erro   : "<<(mlp.erroMedio < eps ? 0 : mlp.erroMedio)<<endl;
        db<<"------------------------"<<endl;
    }
    input.close();
    target.close();
}

void testMLP(string testFile){
    ///le arquivos
    ifstream test (testFile);
    if (!test.is_open()) {cout<<"test deu problema\n"; return;}

    ///inicia teste
    vector<double> inputData;
    vector<double> result;

    string line;
    while(getline(test, line)){

        ///feed forward
        inputData = getInput(line);
        mlp.feedForward(inputData);
        cout<<"Entrada: "; printVector(inputData, true);

        ///obtem resultado atual da rede
        mlp.getResults(result);
        cout<<"Saida  : "; printVector(result, true);
        cout<<"------------------------"<<endl;
    }
    test.close();
}

int main(){
    srand(time(NULL));
    trainMLP("input.txt", "output.txt");
    testMLP("input.txt");
}
