package com.ctuconnect.service;

import com.ctuconnect.dto.BatchDTO;
import com.ctuconnect.entity.BatchEntity;
import com.ctuconnect.repository.BatchRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class BatchService {
    private final BatchRepository batchRepository;

    public List<BatchDTO> getAllBatches() {
        return batchRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public Optional<BatchDTO> getBatchByYear(String year) {
        return batchRepository.findById(year)
                .map(this::convertToDTO);
    }

    public BatchDTO createBatch(BatchDTO batchDTO) {
        BatchEntity batch = BatchEntity.builder()
                .year(batchDTO.getYear())
                .build();
        BatchEntity savedBatch = batchRepository.save(batch);
        return convertToDTO(savedBatch);
    }

    public Optional<BatchDTO> updateBatch(String year, BatchDTO batchDTO) {
        return batchRepository.findById(year)
                .map(existingBatch -> {
                    existingBatch.setYear(batchDTO.getYear());
                    BatchEntity savedBatch = batchRepository.save(existingBatch);
                    return convertToDTO(savedBatch);
                });
    }

    public boolean deleteBatch(String year) {
        if (batchRepository.existsById(year)) {
            batchRepository.deleteById(year);
            return true;
        }
        return false;
    }

    private BatchDTO convertToDTO(BatchEntity batch) {
        return BatchDTO.builder()
                .year(batch.getYear())
                .build();
    }
}
