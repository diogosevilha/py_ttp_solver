# Importando bibliotecas necessárias
import pandas as pd
import re
import math
from amplpy import AMPL, DataFrame

class TTPInstance:
    def __init__(self, arquivo):
        self.arquivo = arquivo
        self.matriz_distancias= None
        self.total_cidades = None
        self.n_item = None
        self.capacity = None
        self.min_speed = None
        self.max_speed = None
        self.rent_ratio = None
        self.df_cidades = None
        self._ler_instancia()

    def _ler_linha_específica(self, numero_linha):
        with open(self.arquivo, 'r') as file:
            linhas = file.readlines()
            if 1 <= numero_linha <= len(linhas):
                return linhas[numero_linha - 1]

    def _ler_arquivo(self):
        with open(self.arquivo, 'r') as file:
            linhas = file.readlines()
        return linhas

    def _extrair_numero(self, linha):
        matches = re.findall(r'\d+\.\d+|\d+', linha)
        if matches:
            return float(matches[-1])
        else:
            raise ValueError(f"Não foi possível extrair um número da linha: {linha}")

    def _ler_instancia(self):
        # Extraindo parâmetros gerais
        self.total_cidades = int(self._extrair_numero(self._ler_linha_específica(3)))
        self.n_item = int(self._extrair_numero(self._ler_linha_específica(4)))
        self.capacity = int(self._extrair_numero(self._ler_linha_específica(5)))
        self.min_speed = self._extrair_numero(self._ler_linha_específica(6))
        self.max_speed = self._extrair_numero(self._ler_linha_específica(7))
        self.rent_ratio = self._extrair_numero(self._ler_linha_específica(8))

        # Lendo as coordenadas das cidades e itens
        linhas = self._ler_arquivo()

        # Encontrar o índice das seções
        inicio_cidades = None
        inicio_itens = None

        for idx, linha in enumerate(linhas):
            linha_strip = linha.strip()
            if linha_strip.startswith('NODE_COORD_SECTION'):
                inicio_cidades = idx + 1  # A próxima linha é o início dos dados das cidades
            elif linha_strip.startswith('ITEMS SECTION'):
                inicio_itens = idx + 1  # A próxima linha é o início dos dados dos itens
                break  # Após encontrar ITEMS SECTION, não precisamos continuar

        if inicio_cidades is None or inicio_itens is None:
            raise ValueError("Não foi possível encontrar as seções NODE_COORD_SECTION ou ITEMS SECTION no arquivo.")

        cidades_dados = linhas[inicio_cidades:inicio_itens - 1]
        itens_dados = linhas[inicio_itens:]

        # Processando dados das cidades
        cidades = []
        for linha in cidades_dados:
            partes = linha.strip().split()
            if len(partes) >= 3:
                index = int(partes[0])
                x = float(partes[1])
                y = float(partes[2])
                cidades.append({'index': index, 'x': x, 'y': y})

        self.df_cidades = pd.DataFrame(cidades)
        self.df_cidades.set_index('index', inplace=True)

        # Processando dados dos itens
        itens = []
        for linha in itens_dados:
            partes = linha.strip().split()
            if len(partes) >= 4:
                index = int(partes[0])
                profit = float(partes[1])
                weight = float(partes[2])
                cidade = int(partes[3])
                itens.append({'index': index, 'profit': profit, 'weight': weight, 'cidade': cidade})

        self.df_itens = pd.DataFrame(itens)
        self.df_itens.set_index('index', inplace=True)

        # Calculando a matriz de distâncias
        self._calcular_matriz_distancias()

    def _calcular_matriz_distancias(self):
        def calcular_distancia(cidade1, cidade2):
            return math.hypot(cidade2['x'] - cidade1['x'], cidade2['y'] - cidade1['y'])

        self.matriz_distancias = {}
        indices = self.df_cidades.index.tolist()
        for i in indices:
            for j in indices:
                if i != j:
                    dist = calcular_distancia(self.df_cidades.loc[i], self.df_cidades.loc[j])
                    self.matriz_distancias[i, j] = dist

class TTPIntegratedSolver:
    def __init__(self, instancia):
        self.instancia = instancia
        self.ampl = AMPL()
        self.ampl.setOption('solver', 'gurobi')
        self._definir_modelo()
        self._definir_dados()

    def _definir_modelo(self):
        modelo_ampl = '''
        set NODES;

        param n integer;
        param m integer;
        param W;  # Capacidade máxima da mochila
        param vmax;
        param vmin;
        param R;

        param Profit{1..m};
        param Weight{1..m};
        param AssignNode{1..m};

        param Distance{NODES, NODES};

        var x{NODES, NODES} binary;
        var y{1..m} binary;
        var u{NODES} >= 0;

        minimize TotalCost:
            sum {i in NODES, j in NODES : i != j} (Distance[i,j] / (vmax - (vmax - vmin) * (sum {k in 1..m : AssignNode[k] == i} Weight[k] * y[k]) / W)) * x[i,j]
            - sum {k in 1..m} Profit[k] * y[k] + R * sum {i in NODES, j in NODES : i != j} (Distance[i,j] / (vmax - (vmax - vmin) * (sum {k in 1..m : AssignNode[k] == i} Weight[k] * y[k]) / W)) * x[i,j];

        subject to OneOut{i in NODES}:
            sum {j in NODES : j != i} x[i,j] == 1;

        subject to OneIn{j in NODES}:
            sum {i in NODES : i != j} x[i,j] == 1;

        subject to SubtourElimination{i in NODES, j in NODES : i != j and i != 1 and j != 1}:
            u[i] - u[j] + n * x[i,j] <= n - 1;

        subject to KnapsackConstraint:
            sum {k in 1..m} Weight[k] * y[k] <= W;
        '''
        self.ampl.eval(modelo_ampl)

    def _definir_dados(self):
        ampl = self.ampl
        instancia = self.instancia

        ampl.set['NODES'] = instancia.df_cidades.index.tolist()
        ampl.param['n'] = instancia.total_cidades
        ampl.param['m'] = instancia.n_item
        ampl.param['W'] = instancia.capacity
        ampl.param['vmax'] = instancia.max_speed
        ampl.param['vmin'] = instancia.min_speed
        ampl.param['R'] = instancia.rent_ratio

        # Definindo parâmetros dos itens
        ampl.param['Profit'] = instancia.df_itens['profit'].to_dict()
        ampl.param['Weight'] = instancia.df_itens['weight'].to_dict()
        ampl.param['AssignNode'] = instancia.df_itens['cidade'].to_dict()

        # Definindo a matriz de distâncias
        ampl.param['Distance'] = instancia.matriz_distancias

    def solve(self):
        self.ampl.solve()
        self._extrair_resultados()

    def _extrair_resultados(self):
        ampl = self.ampl

        x_sol = ampl.getVariable('x').getValues()
        y_sol = ampl.getVariable('y').getValues()

        # Convertendo os valores de x e y para dicionários
        x_values = x_sol.toDict()
        y_values = y_sol.toDict()

        # Reconstruindo a rota
        rota = []
        current_city = 1  # Supondo que a cidade inicial é a de índice 1
        visited = set()
        while True:
            visited.add(current_city)
            # Obter as cidades adjacentes a current_city com x[current_city, j] > 0.5
            next_cities = [j for j in self.instancia.df_cidades.index if x_values.get((current_city, j), 0) > 0.5]
            if not next_cities or next_cities[0] in visited:
                break
            next_city = next_cities[0]
            rota.append((current_city, next_city))
            current_city = next_city

        self.rota = rota

        # Itens selecionados
        itens_selecionados = [k for k in self.instancia.df_itens.index if y_values.get(k, 0) > 0.5]
        self.itens_selecionados = itens_selecionados

        # Lucro total
        self.lucro_total = ampl.getObjective('TotalCost').value()

    def resultados(self):
        print("===== TTP Integrado =====")
        print("Rota Ótima:")
        for (i, j) in self.rota:
            print(f"Cidade {i} -> Cidade {j}")

        print("\nItens Selecionados:")
        for k in self.itens_selecionados:
            item = self.instancia.df_itens.loc[k]
            print(f"Item {k}: Lucro = {item['profit']}, Peso = {item['weight']}, Cidade = {item['cidade']}")

        print(f"\nLucro Total: {self.lucro_total}")

class TSPThenKPSolver:
    def __init__(self, instancia):
        self.instancia = instancia
        self.ampl = AMPL()
        self.ampl.setOption('solver', 'gurobi')

    def solve(self):
        self._resolver_tsp()
        self._resolver_kp()
        self._calcular_ttp()

    def _resolver_tsp(self):
        ampl = self.ampl

        modelo_tsp = '''
        set NODES;

        param n integer;
        param Distance{NODES, NODES};

        var x{NODES, NODES} binary;
        var u{NODES} >= 0;

        minimize TotalDistance:
            sum {i in NODES, j in NODES : i != j} Distance[i,j] * x[i,j];

        subject to OneOut{i in NODES}:
            sum {j in NODES : j != i} x[i,j] == 1;

        subject to OneIn{j in NODES}:
            sum {i in NODES : i != j} x[i,j] == 1;

        subject to SubtourElimination{i in NODES, j in NODES : i != j and i != 1 and j != 1}:
            u[i] - u[j] + n * x[i,j] <= n - 1;
        '''
        ampl.eval(modelo_tsp)

        ampl.set['NODES'] = self.instancia.df_cidades.index.tolist()
        ampl.param['n'] = self.instancia.total_cidades
        ampl.param['Distance'] = self.instancia.matriz_distancias

        ampl.solve()

        x_sol = ampl.getVariable('x').getValues()
        x_values = x_sol.toDict()

        # Reconstruindo a rota
        rota = []
        current_city = 1  # Supondo que a cidade inicial é a de índice 1
        visited = set()
        while True:
            visited.add(current_city)
            next_cities = [j for j in self.instancia.df_cidades.index if x_values.get((current_city, j), 0) > 0.5]
            if not next_cities or next_cities[0] in visited:
                break
            next_city = next_cities[0]
            rota.append((current_city, next_city))
            current_city = next_city

        self.rota = rota

    def _resolver_kp(self):
        ampl = AMPL()
        ampl.setOption('solver', 'gurobi')

        modelo_kp = '''
        param m integer;
        param W;

        param Profit{1..m};
        param Weight{1..m};

        var y{1..m} binary;

        maximize TotalProfit:
            sum {k in 1..m} Profit[k] * y[k];

        subject to KnapsackConstraint:
            sum {k in 1..m} Weight[k] * y[k] <= W;
        '''
        ampl.eval(modelo_kp)

        instancia = self.instancia

        ampl.param['m'] = instancia.n_item
        ampl.param['W'] = instancia.capacity
        ampl.param['Profit'] = instancia.df_itens['profit'].to_dict()
        ampl.param['Weight'] = instancia.df_itens['weight'].to_dict()

        ampl.solve()

        y_sol = ampl.getVariable('y').getValues()
        y_values = y_sol.toDict()

        itens_selecionados = [k for k in instancia.df_itens.index if y_values.get(k, 0) > 0.5]

        self.itens_selecionados = itens_selecionados

    def _calcular_ttp(self):
        # Calcular lucro total considerando a rota e itens selecionados
        instancia = self.instancia

        # Peso acumulado
        peso_acumulado = 0
        tempo_total = 0
        lucro_itens = 0

        for (i, j) in self.rota:
            # Adiciona itens coletados na cidade i
            itens_na_cidade = [k for k in self.itens_selecionados if instancia.df_itens.loc[k]['cidade'] == i]
            peso_cidade = sum(instancia.df_itens.loc[k]['weight'] for k in itens_na_cidade)
            lucro_cidade = sum(instancia.df_itens.loc[k]['profit'] for k in itens_na_cidade)
            peso_acumulado += peso_cidade
            lucro_itens += lucro_cidade

            # Calcula a velocidade atual
            velocidade = instancia.max_speed - (instancia.max_speed - instancia.min_speed) * (peso_acumulado / instancia.capacity)

            # Tempo para percorrer de i para j
            distancia = instancia.matriz_distancias[i, j]
            tempo = distancia / velocidade
            tempo_total += tempo

        custo_aluguel = instancia.rent_ratio * tempo_total

        self.lucro_total = lucro_itens - custo_aluguel

    def resultados(self):
        print("===== TSP -> KP =====")
        print("Rota Ótima:")
        for (i, j) in self.rota:
            print(f"Cidade {i} -> Cidade {j}")

        print("\nItens Selecionados:")
        for k in self.itens_selecionados:
            item = self.instancia.df_itens.loc[k]
            print(f"Item {k}: Lucro = {item['profit']}, Peso = {item['weight']}, Cidade = {item['cidade']}")

        print(f"\nLucro Total: {self.lucro_total}")

class KPThenTSPSolver:
    def __init__(self, instancia):
        self.instancia = instancia
        self.ampl = AMPL()
        self.ampl.setOption('solver', 'gurobi')

    def solve(self):
        self._resolver_kp()
        self._resolver_tsp()
        self._calcular_ttp()

    def _resolver_kp(self):
        ampl = AMPL()
        ampl.setOption('solver', 'gurobi')

        modelo_kp = '''
        param m integer;
        param W;

        param Profit{1..m};
        param Weight{1..m};

        var y{1..m} binary;

        maximize TotalProfit:
            sum {k in 1..m} Profit[k] * y[k];

        subject to KnapsackConstraint:
            sum {k in 1..m} Weight[k] * y[k] <= W;
        '''
        ampl.eval(modelo_kp)

        instancia = self.instancia

        ampl.param['m'] = instancia.n_item
        ampl.param['W'] = instancia.capacity
        ampl.param['Profit'] = instancia.df_itens['profit'].to_dict()
        ampl.param['Weight'] = instancia.df_itens['weight'].to_dict()

        ampl.solve()

        y_sol = ampl.getVariable('y').getValues()
        y_values = y_sol.toDict()

        itens_selecionados = [k for k in instancia.df_itens.index if y_values.get(k, 0) > 0.5]

        self.itens_selecionados = itens_selecionados

    def _resolver_tsp(self):
        ampl = self.ampl

        # Considerar apenas cidades com itens selecionados
        cidades_itens = set(self.instancia.df_itens.loc[self.itens_selecionados]['cidade'])
        cidades_tsp = [1] + list(cidades_itens)  # Incluir a cidade inicial

        modelo_tsp = '''
        set NODES;

        param n integer;
        param Distance{NODES, NODES};

        var x{NODES, NODES} binary;
        var u{NODES} >= 0;

        minimize TotalDistance:
            sum {i in NODES, j in NODES : i != j} Distance[i,j] * x[i,j];

        subject to OneOut{i in NODES}:
            sum {j in NODES : j != i} x[i,j] == 1;

        subject to OneIn{j in NODES}:
            sum {i in NODES : i != j} x[i,j] == 1;

        subject to SubtourElimination{i in NODES, j in NODES : i != j and i != 1 and j != 1}:
            u[i] - u[j] + n * x[i,j] <= n - 1;
        '''
        ampl.eval(modelo_tsp)

        ampl.set['NODES'] = cidades_tsp
        ampl.param['n'] = len(cidades_tsp)

        # Filtrar a matriz de distâncias
        matriz_distancias_tsp = {(i, j): self.instancia.matriz_distancias[i, j] for i in cidades_tsp for j in cidades_tsp if i != j}
        ampl.param['Distance'] = matriz_distancias_tsp

        ampl.solve()

        x_sol = ampl.getVariable('x').getValues()
        x_values = x_sol.toDict()

        # Reconstruindo a rota
        rota = []
        current_city = 1  # Cidade inicial
        visited = set()
        while True:
            visited.add(current_city)
            next_cities = [j for j in cidades_tsp if x_values.get((current_city, j), 0) > 0.5]
            if not next_cities or next_cities[0] in visited:
                break
            next_city = next_cities[0]
            rota.append((current_city, next_city))
            current_city = next_city

        self.rota = rota

    def _calcular_ttp(self):
        # Calcular lucro total considerando a rota e itens selecionados
        instancia = self.instancia

        # Peso acumulado
        peso_acumulado = 0
        tempo_total = 0
        lucro_itens = 0

        for (i, j) in self.rota:
            # Adiciona itens coletados na cidade i
            itens_na_cidade = [k for k in self.itens_selecionados if instancia.df_itens.loc[k]['cidade'] == i]
            peso_cidade = sum(instancia.df_itens.loc[k]['weight'] for k in itens_na_cidade)
            lucro_cidade = sum(instancia.df_itens.loc[k]['profit'] for k in itens_na_cidade)
            peso_acumulado += peso_cidade
            lucro_itens += lucro_cidade

            # Calcula a velocidade atual
            velocidade = instancia.max_speed - (instancia.max_speed - instancia.min_speed) * (peso_acumulado / instancia.capacity)

            # Tempo para percorrer de i para j
            distancia = instancia.matriz_distancias[i, j]
            tempo = distancia / velocidade
            tempo_total += tempo

        custo_aluguel = instancia.rent_ratio * tempo_total

        self.lucro_total = lucro_itens - custo_aluguel

    def resultados(self):
        print("===== KP -> TSP =====")
        print("Rota Ótima:")
        for (i, j) in self.rota:
            print(f"Cidade {i} -> Cidade {j}")

        print("\nItens Selecionados:")
        for k in self.itens_selecionados:
            item = self.instancia.df_itens.loc[k]
            print(f"Item {k}: Lucro = {item['profit']}, Peso = {item['weight']}, Cidade = {item['cidade']}")

        print(f"\nLucro Total: {self.lucro_total}")

# Exemplo de uso
if __name__ == "__main__":
    arquivo_instancia = 'data/eil51_n50_bounded-strongly-corr_01.txt'
    instancia = TTPInstance(arquivo_instancia)

    # Resolver o TTP integrado
    #solver_integrado = TTPIntegratedSolver(instancia)
    #solver_integrado.solve()
    #solver_integrado.resultados()

    print("\n----------------------------\n")

    # Resolver primeiro o TSP, depois o KP
    solver_tsp_kp = TSPThenKPSolver(instancia)
    print("inicio TSPThenKPSolver")
    solver_tsp_kp.solve()
    print("Finalizado TSPThenKPSolver")
    solver_tsp_kp.resultados()

    print("\n----------------------------\n")

    # Resolver primeiro o KP, depois o TSP
    solver_kp_tsp = KPThenTSPSolver(instancia)
    solver_kp_tsp.solve()
    solver_kp_tsp.resultados()
