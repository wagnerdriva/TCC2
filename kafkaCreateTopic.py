from kafka.admin import KafkaAdminClient, NewTopic
import sys

admin_client = KafkaAdminClient(
    bootstrap_servers="192.168.200.110:9092", 
    client_id='favorite_group'
)

topicName = sys.argv[1]
if not topicName:
    print("Passe o nome do tópico como parametro...")
    exit()

print(f"Criando topico {topicName}")

topicList = []
topicList.append(NewTopic(name=topicName, num_partitions=1, replication_factor=1))
admin_client.create_topics(new_topics=topicList, validate_only=False)