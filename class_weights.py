''' Script to maintain class weights for different data sets for varying number of classes'''

def get_salumn1_class_weights(n_classes=2):

    '''Returns class weights for SALUMN1 dataset

        @Arguments:
            n_classes: Number of classes for which the weights are required
        @Returns:
            class weights as a list
    '''
    salum1_class_weights = {}

    salum1_class_weights[19] = [6.911896233069708e-06, 0.000937101609474355,
                                0.02345263423380346, 0.5760026776515352, 1.8841621669966422,
                                1.2463678176352848, 2.216279418890707,
                                3.8356630755349665, 3.226099404041575,
                                3.794459435225647, 0.022635901425900756,
                                0.09695134817971188, 1.3980743605839152,
                                0.042602750221501104, 4.028508557841384,
                                3.1631006527711407, 15.509242088692064,
                                16.22600756423384, 0.006645944202755768]

    salum1_class_weights[9] = [0.03248902, 0.97143194, 0.99658679, 0.99961234,
                               0.99999777, 0.99992129, 0.99997822, 0.99999907, 0.99998356]

    salum1_class_weights[4] = [0.03253065, 0.97143825, 0.99647538, 0.99955572]

    salum1_class_weights[2] = [0.05, 0.95]

    return salum1_class_weights[n_classes]


def get_gap_class_weights(n_classes=2):
    '''Returns class weights for GAP dataset

        @Arguments:
            n_classes: Number of classes for which the weights are required
        @Returns:
            class weights as a list
    '''
    gap_class_weights = {}

    gap_class_weights[2] = [0.19, 0.81]

    return gap_class_weights[n_classes]
    