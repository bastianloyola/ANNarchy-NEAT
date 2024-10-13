import json
bestFile = 'best0.txt'
configFile = 'config.cfg'
infoFile = 'info.txt'
operadoresFile = 'operadores.txt'
resultsFile = 'results.txt'

def leerBestFile(file, nodos, conexiones):
    with open(file, 'r') as best:
        for line in best:
            line = line.strip().split(';')
            nodos.add(int(line[0]))
            nodos.add(int(line[1]))
            conexiones.add((int(line[0]), int(line[1])))

def leerConfigFile(file, configuraciones):
    with open(file, 'r') as config_file:
        for line in config_file:
            line = line.strip().split('=')
            if len(line) == 2:
                key = line[0].strip()
                value = line[1].strip()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                configuraciones[key] = value

def leerOperadoresFile(file, operadores):
    print('leyendo ' + file)
    with open(file, 'r') as op:
        for line in op:
            if line == '\n' or 'Generacion' in line:
                continue
            if 'Total' in line:
                break
            line = line.strip().split()
            if len(line) == 3:
                key = line[1].strip()[:-1]
                num = int(line[2].strip())
                print(key, num)
                if key in operadores:
                    operadores[key].append(num)

def leerInfoFile(file, info):
    gen = 0
    etapa = ''
    with open(file, 'r') as info_file:
        for line in info_file:
            if 'Evaluation' in line:
                etapa = 'Evaluation'
                print(etapa)
                continue
            elif 'Eliminate' in line:
                etapa = 'Eliminate'
                print(etapa)
                continue
            elif ' Mutation' in line:
                etapa = 'Mutation'
                print(etapa)
                continue
            elif 'Reproduce' in line:
                etapa = 'Reproduce'
                print(etapa)
                continue
            elif 'Speciation' in line:
                etapa = 'Speciation'
                print(etapa)
                continue
            elif 'Best Genome' in line:
                etapa = 'BestGenome'
                print(etapa)
                continue
            elif 'Generation' in line:
                print(line)
                line = line.strip().split()
                gen = int(line[2])
                info['reproducidos'].append([0, 0, 0])
                info['species'].append([])
                info['bestGenome'].append([])
                info['eliminados'].append(0)
                continue
            elif line == '\n':
                continue

            elif etapa == 'Eliminate':
                if 'eliminated' in line:
                    info['eliminados'][gen] += 1
                continue
            elif etapa == 'Mutation':
                continue
            elif etapa == 'Reproduce':
                line = line.strip().split()
                if 'reproduceInterSpecies' in line[1]:
                    info['reproducidos'][gen][0] += int(line[2])
                elif 'reproduceNonInterSpecies' in line[1]:
                    info['reproducidos'][gen][1] += int(line[2])
                elif 'reproduceMutations' in line[1]:
                    info['reproducidos'][gen][2] += int(line[2])
                continue
            elif etapa == 'Speciation':
                line = line.strip().split()
                if 'Species' in line[0]:
                    info['species'][gen].append(int(line[3]))
                continue
            elif etapa == 'BestGenome':
                line = line.strip().split(':')
                num = float(line[1]) if 'fitness' in line[0] else int(line[1])
                info['bestGenome'][gen].append(num)
                print(num)
                continue


def leerResultsFile(file, info):
    with open(file, 'r') as results_file:
        for line in results_file:
            if 'Genome fitness' in line:
                line = line.strip().split()
                num = float(line[2])
                info['bestGenome'][-1].append(0)
                info['bestGenome'][-1].append(0)
                info['bestGenome'][-1].append(num)


carpetas = ['A-IZ/','A-LIF/'] #MODIFICAR
trials_por_carpeta = [11,11] #MODIFICAR

for k in range(len(carpetas)):
    for j in range(trials_por_carpeta[k]):
        file = carpetas[k]+'trial-'+str(j+1)+'/'
        print(file)

        nodos = set()
        conexiones = set()
        configuraciones = dict()
        operadores = {'mutacionPeso': [], 'mutacionPesoInput': [], 'agregarNodos': [], 'agregarLinks': [], 'reproducirInter': [], 'reproducirIntra': [], 'reproducirMuta': []}
        info = {'eliminados': [], 'reproducidos': [], 'species': [], 'bestGenome': []}

        leerConfigFile(file + configFile, configuraciones)
        #print('Configuracion: \n', configuraciones)
        leerOperadoresFile(file + operadoresFile, operadores)
        #print('Operadores: \n', operadores)
        leerBestFile(file + bestFile, nodos, conexiones)
        #print('Best: \n', nodos, conexiones)
        leerInfoFile(file + infoFile, info)
        #print('Info: \n', info)
        leerResultsFile(file + resultsFile, info)

        #print('Info: ')
        #for i in range(configuraciones.get('evolutions', 0)):
            #print(i)
            #print('--> Eliminados: ', info['eliminados'][i])
            #print('--> Reproducidos: ', info['reproducidos'][i])
            #print('--> Species: ', info['species'][i])
            #print('--> BestGenome: ', info['bestGenome'][i])

        

        outputFile = file+'output.json'

        data_to_save = {
            'Configuracion': configuraciones,
            'Operadores': operadores,
            'Nodos': list(nodos),
            'Conexiones': list(conexiones),
            'Info': {
                'Eliminados': info['eliminados'],
                'Reproducidos': info['reproducidos'],
                'Species': info['species'],
                'BestGenome': info['bestGenome']
            }
        }

        with open(outputFile, 'w') as f:
            json.dump(data_to_save, f, indent=4, separators=(", ", ": "))

        print(f"Datos guardados en {outputFile}")